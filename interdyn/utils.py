from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import os
import random

import av  # noqa: F401  # imported for side‑effects in some environments
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from torchvision import tv_tensors
import torchvision.transforms.v2 as T
from torchvision.io import write_video
from torchvision.transforms.v2 import Transform
import torchvision.transforms.v2.functional as F
from torchvision.transforms.v2 import InterpolationMode

import decord
from decord import VideoReader, cpu
decord.bridge.set_bridge("torch")


class MaskAwareCrop(nn.Module):
    """
    Crop a video and its corresponding mask so that the cropped region
    stays centered on the area where the mask is “visible” (i.e., nonzero).
    If the mask is empty, falls back to a center crop.

    Args:
        size (int or (int, int)): Desired output size (height, width).
    """

    def __init__(self, size: Union[int, Sequence[int]]) -> None:
        super().__init__()
        if isinstance(size, int):
            self.height: int = size
            self.width: int = size
        else:
            self.height, self.width = size

    def forward(
        self,
        video: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            masks (Tensor): shape [..., H, W] or [T, C, H, W], where any nonzero
                pixel indicates “visible” region to focus on.
            video (Tensor): same spatial dimensions as `masks`.

        Returns:
            cropped_masks (Tensor): the mask tensor cropped to (height, width).
            cropped_video (Tensor): the video tensor cropped to (height, width).

        Behavior:
        1. Compute the bounding box of all nonzero mask pixels across all
           leading dimensions.
        2. Compute its center (cx, cy) and then place a (height × width)
           window around that center, clamped to image borders.
        3. If the mask is all zero, default to a center crop.
        """
        # Find any nonzero across all dimensions except the last two (H, W)
        pos_any = (masks > 0).any(dim=tuple(range(masks.ndim - 2)))
        ys, xs = torch.where(pos_any)

        H, W = masks.shape[-2], masks.shape[-1]

        if ys.numel() == 0:
            # No mask: center crop
            top = (H - self.height) // 2
            left = (W - self.width) // 2
        else:
            # Compute bounding box of visible region
            min_y, max_y = int(ys.min()), int(ys.max())
            min_x, max_x = int(xs.min()), int(xs.max())
            cy = (min_y + max_y) // 2
            cx = (min_x + max_x) // 2

            # Place crop window centered on (cy, cx), clamped to image bounds
            top = min(max(0, cy - self.height // 2), H - self.height)
            left = min(max(0, cx - self.width  // 2), W - self.width)

        cropped_video = F.crop(
            video,
            top=top,
            left=left,
            height=self.height,
            width=self.width,
        )

        cropped_masks = F.crop(
            masks,
            top=top,
            left=left,
            height=self.height,
            width=self.width,
        )

        return cropped_video, cropped_masks


class Upsample:
    """
    Resize an image or video so that its shorter side meets or exceeds the
    target dimensions, preserving aspect ratio.

    Args:
        target_height (int): Minimum height in pixels after resizing.
        target_width (int): Minimum width in pixels after resizing.
        interpolation (InterpolationMode, optional): Interpolation mode to use.
            Defaults to InterpolationMode.BICUBIC.

    Call:
        upsample = Upsample(256, 256)
        output = upsample(frame)

    Where:
        frame (Tensor): `[C, H, W]` or `[T, C, H, W]`. Output will have the
            same rank, dtype, and device as input.
    """

    def __init__(
        self,
        target_height: int,
        target_width: int,
        interpolation: InterpolationMode = InterpolationMode.BICUBIC,
    ) -> None:
        self.target_height: int = target_height
        self.target_width: int = target_width
        self.interpolation: InterpolationMode = interpolation

    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Compute scale factor so that both height and width are at least
        the target, then resize.

        Args:
            frame (Tensor): Input image `[C, H, W]` or video `[T, C, H, W]`.

        Returns:
            Tensor: Resized tensor with same number of dims as input.
        """
        # Get spatial dimensions
        h, w = frame.shape[-2], frame.shape[-1]
        # Determine scaling factor
        scale = max(self.target_height / h, self.target_width / w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        # Resize while preserving aspect ratio
        return F.resize(
            frame,
            size=[new_h, new_w],
            interpolation=self.interpolation,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"min_height={self.target_height}, "
            f"min_width={self.target_width}, "
            f"interpolation={self.interpolation.value})"
        )


def load_frames(video_path: str, indices: Sequence[int]) -> tv_tensors.Video:
    """
    Read frames at given indices from a video file.
    """
    video_reader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    video = video_reader.get_batch(indices).permute(0, 3, 1, 2)
    return tv_tensors.Video(video)


def sample_frames(
    start_indices,
    min_num_frames,
    resample_factor,
) -> List[int]:
    """
    Randomly choose a start and return spaced indices.
    """
    start_idx = random.choice(start_indices)
    return list(range(start_idx, start_idx + min_num_frames, resample_factor))


def format(
    video: torch.Tensor,
    data_range,
    data_format,
) -> torch.Tensor:
    """
    Reorder channels and normalize pixel range.
    """
    if data_format == "THWC":
        video = video.permute(0, 2, 3, 1)

    if data_range == "0-255":
        return video

    video = video / 255.0
    if data_range == "0-1":
        return video

    return video * 2.0 - 1.0


def build_transform(
    crop_type,
    target_height,
    target_width,
    generator,
) -> Optional[T.Compose]:
    """
    Build a joint transform pipeline for video and mask inputs,
    including resize, crop, and optional augmentations.
    """
    def branch(idx: int, transform: Transform) -> Transform:
        # Apply `t` only to the idx-th input in a tuple
        return lambda *imgs: tuple(
            transform(img) if i == idx else img
            for i, img in enumerate(imgs)
        )

    transforms = []

    # Resize video (bicubic) and masks (nearest)
    transforms.append(branch(0, Upsample(target_height, target_width, InterpolationMode.BICUBIC)))
    transforms.append(branch(1, Upsample(target_height, target_width, InterpolationMode.NEAREST)))

    # Crop strategy
    if crop_type == "maximize_visibility":
        transforms.append(MaskAwareCrop((target_height, target_width)))
    elif crop_type == "random":
        transforms.append(T.RandomCrop((target_height, target_width), generator=generator))
    else:
        transforms.append(T.CenterCrop((target_height, target_width)))

    return T.Compose(transforms) if transforms else None


def load_sample(
    local_dir,
    generator,
    crop_type: str = "center",
    target_fps: int = 6,
    target_num_frames: int = 14,
    target_height: int = 256,
    target_width: int = 384,
    first_frame_visibility: bool = True,
    continuous_visibility: bool = False,
    data_range: str = "0-1",
    data_format: str = "TCHW",
) -> None:

    transform = build_transform(
        crop_type=crop_type,
        target_height=target_height,
        target_width=target_width,
        generator=generator,
    )

    video_path = f"{local_dir}/video.webm"
    control_path = f"{local_dir}/mask_hand.webm"

    vr = VideoReader(control_path, ctx=cpu(0), num_threads=1)
    num_frames, fps = len(vr), int(vr.get_avg_fps())

    resample_factor = (fps // target_fps)
    min_num_frames = target_num_frames * resample_factor

    assert fps % target_fps == 0, "Video FPS must be divisible by target FPS"
    assert fps >= target_fps, "Video FPS must be larger than target FPS"
    assert num_frames >= min_num_frames, "Not enough frames in video"

    # Sample frame indices
    indices = sample_frames(
        start_indices=list(range(num_frames - min_num_frames + 1)),
        min_num_frames=min_num_frames,
        resample_factor=resample_factor,
    )

    # Load sequences
    frames = load_frames(video_path, indices)
    control = load_frames(control_path, indices)

    # Joint transform: frames and masks
    frames, control = transform(frames, control)
    frames_t = frames.as_subclass(torch.Tensor)
    control_t = control.as_subclass(torch.Tensor)

    # Reformat channels and scale
    frames_t = format(frames_t, data_range, data_format).unsqueeze(0)
    control_t = format(control_t, data_range, data_format).unsqueeze(0)

    return frames_t, control_t


def post_process_sample(frames, conds, preds):

    def _scale(x: torch.Tensor) -> torch.Tensor:
        return (x * 255).to(torch.uint8)

    def _annotate(seq: torch.Tensor, tag: str) -> torch.Tensor:

        """Draw `tag` in the top-right corner of each (C,H,W) frame tensor."""

        font = ImageFont.load_default()
        out = []

        for frame in seq:

            img = F.to_pil_image(frame)
            draw = ImageDraw.Draw(img)

            x0, y0, x1, y1 = draw.textbbox((0, 0), tag, font=font)
            tw, th = x1 - x0, y1 - y0
            w, h = img.size

            draw.rectangle([(w - tw - 6, 0), (w, th + 4)], fill=(255, 255, 255))
            draw.text((w - tw - 3, 2), tag, fill=(0, 0, 0), font=font)
            out.append(F.pil_to_tensor(img))

        return torch.stack(out)

    frames = _scale(frames.squeeze(0).cpu())
    conds = _scale(conds.squeeze(0).cpu())
    preds = [_scale(pred.cpu()) for pred in preds]

    num_frames = frames.shape[0]
    input = frames[:1].repeat(num_frames, 1, 1, 1)

    input = _annotate(input, "IN")
    conds = _annotate(conds, "COND")
    preds = [_annotate(pred, f"P{idx+1}") for idx, pred in enumerate(preds)]
    frames_gt = _annotate(frames, "GT")

    comparison = torch.cat([input, conds, *preds, frames_gt], dim=3)

    video_dict = {
        "ground_truth": frames_gt,
        "conditioning": conds,
        "comparison": comparison,
    }
    for idx, pred in enumerate(preds):
        video_dict[f"prediction_{idx}"] = pred

    return video_dict


def log_local(video_dict, log_path, id, fps=6):

    video_dir = os.path.join(log_path, id)
    os.makedirs(video_dir, exist_ok=True)

    for name, video in video_dict.items():
        
        filename = os.path.join(video_dir, f"{name}.mp4")
        video_array = video.permute(0, 2, 3, 1).contiguous()

        write_video(
            filename=filename,
            video_array=video_array,
            fps=fps,
            video_codec='libx264',
            options={'crf': '18'}
        )