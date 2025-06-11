from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import PIL.Image
from transformers import CLIPVisionModelWithProjection
from transformers import CLIPImageProcessor

from diffusers.image_processor import PipelineImageInput
from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler

from diffusers.utils import is_torch_xla_available, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    _append_dims,
    retrieve_timesteps,
    _resize_with_antialiasing,
    StableVideoDiffusionPipelineOutput,
    StableVideoDiffusionPipeline
)

from .video_processor import VideoProcessor
from .controlnet_sdv import ControlNetSDVModel
from .unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)


class InterDynPipeline(StableVideoDiffusionPipeline):

    # model_cpu_offload_seq = "image_encoder->unet->vae"
    # _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNetSpatioTemporalConditionControlNetModel,
        scheduler: EulerDiscreteScheduler,
        feature_extractor: CLIPImageProcessor,
        controlnet: ControlNetSDVModel,
    ):

        super().__init__(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            controlnet=controlnet,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(
            do_resize=True,
            vae_scale_factor=self.vae_scale_factor
        )

    @torch.inference_mode()
    def _encode_image(
        self,
        image: PipelineImageInput,
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ) -> torch.Tensor:

        dtype = next(self.image_encoder.parameters()).dtype

        image = _resize_with_antialiasing(image, (224, 224))
        image = (image + 1.0) / 2.0

        image = self.feature_extractor(
            images=image,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        return image_embeddings

    @torch.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.Tensor],
        controlnet_cond: torch.Tensor = None,
        controlnet_cond_scale: float = 1.0,
        height: int = 256,
        width: int = 384,
        num_frames: Optional[int] = 14,
        num_inference_steps: int = 25,
        sigmas: Optional[List[float]] = None,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 6,
        motion_bucket_id: int = 40,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
    ):

        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        device = self._execution_device
        self._guidance_scale = max_guidance_scale

        self.check_inputs(image, height, width)

        image = self.video_processor.preprocess_video(
            image, height=height, width=width).squeeze(0).to(device)
        batch_size, num_channels, height, width = image.shape
        
        image_embeddings = self._encode_image(image, device, num_videos_per_prompt, self.do_classifier_free_guidance)

        noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
        image = image + noise_aug_strength * noise

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        image_latents = self._encode_vae_image(
            image,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
        image_latents = image_latents.to(image_embeddings.dtype)

        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            None,
            sigmas
        )

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents)

        controlnet_cond = self.video_processor.preprocess_video(
            controlnet_cond, height=height, width=width)
        controlnet_cond = controlnet_cond.repeat(num_videos_per_prompt, 1, 1, 1, 1)
        controlnet_cond = torch.cat([controlnet_cond] * 2)
        controlnet_cond = controlnet_cond.to(device, latents.dtype)

        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)
        self._guidance_scale = guidance_scale

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=image_embeddings,
                    controlnet_cond=controlnet_cond,
                    added_time_ids=added_time_ids,
                    conditioning_scale=controlnet_cond_scale,
                    guess_mode=False,
                    return_dict=False)

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=image_embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    added_time_ids=added_time_ids,
                    return_dict=False)[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if not output_type == "latent":

            if needs_upcasting:
                self.vae.to(dtype=torch.float16)

            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = self.video_processor.postprocess_video(video=frames, output_type=output_type)

        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)