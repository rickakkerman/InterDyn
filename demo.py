import os
import torch
import argparse
from diffusers.training_utils import set_seed

from interdyn.unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel
from interdyn.controlnet_sdv import ControlNetSDVModel
from interdyn.pipeline import InterDynPipeline
from interdyn.utils import load_sample, post_process_sample, log_local


def demo(args):

    generator = set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=torch.float16,
    )

    controlnet = ControlNetSDVModel.from_pretrained(
        args.controlnet_path,
        subfolder="controlnet",
        torch_dtype=torch.float16,
    )

    pipeline = InterDynPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")

    example_ids = sorted(os.listdir(args.input_dir))

    if args.id == "all":
        ids = example_ids
    elif args.id in example_ids:
        ids = [args.id]
    else:
        raise ValueError(
            f"ID '{args.id}' was not found in '{args.input_dir}'. "
            f"Available IDs: {', '.join(example_ids)}"
        )

    for id in ids:

        frames, controlnet_cond = load_sample(os.path.join(args.input_dir, id), generator=generator)

        pred = pipeline(
            image=frames[:, 0],
            controlnet_cond=controlnet_cond,
            num_inference_steps=args.num_inference_steps,
            num_videos_per_prompt=args.num_videos_per_prompt,
            generator=generator,
            output_type="pt",
        ).frames

        video_dict = post_process_sample(frames, controlnet_cond, pred)
        log_local(video_dict, args.output_dir, str(id), 6)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="examples")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--id", type=str, default="all")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-video-diffusion-img2vid")
    parser.add_argument("--controlnet_path", type=str, default="rickakkerman/InterDyn")
    parser.add_argument("--num_videos_per_prompt", type=int, default=3)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    demo(args)