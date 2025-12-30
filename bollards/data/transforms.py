from __future__ import annotations

from typing import Optional

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from bollards.constants import IMAGENET_MEAN, IMAGENET_STD
from bollards.pipelines.train.config import AugmentConfig


def build_transforms(
    train: bool,
    img_size: int,
    augment: Optional[AugmentConfig] = None,
) -> transforms.Compose:
    if train and augment and augment.enabled:
        resize_pad = max(0, int(augment.resize_pad))
        resize_size = img_size + resize_pad
        crop_min = float(augment.crop_scale_min)
        crop_max = float(augment.crop_scale_max)
        if crop_max < crop_min:
            crop_max = crop_min

        ops = [
            transforms.Resize((resize_size, resize_size), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop(
                img_size,
                scale=(crop_min, crop_max),
                interpolation=InterpolationMode.BICUBIC,
            ),
        ]

        if augment.affine_p > 0:
            translate = None
            if augment.affine_translate and augment.affine_translate > 0:
                translate = (float(augment.affine_translate), float(augment.affine_translate))

            scale = None
            if augment.affine_scale_min or augment.affine_scale_max:
                scale = (float(augment.affine_scale_min), float(augment.affine_scale_max))

            affine = transforms.RandomAffine(
                degrees=float(augment.affine_degrees),
                translate=translate,
                scale=scale,
                interpolation=InterpolationMode.BICUBIC,
                fill=0,
            )
            ops.append(transforms.RandomApply([affine], p=float(augment.affine_p)))

        if augment.hflip_p > 0:
            ops.append(transforms.RandomHorizontalFlip(p=float(augment.hflip_p)))

        if any([augment.brightness, augment.contrast, augment.saturation, augment.hue]):
            ops.append(
                transforms.ColorJitter(
                    brightness=float(augment.brightness),
                    contrast=float(augment.contrast),
                    saturation=float(augment.saturation),
                    hue=float(augment.hue),
                )
            )

        if augment.sharpness_p > 0:
            ops.append(
                transforms.RandomAdjustSharpness(
                    sharpness_factor=float(augment.sharpness_factor),
                    p=float(augment.sharpness_p),
                )
            )

        if augment.blur_p > 0:
            ops.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=int(augment.blur_kernel))],
                    p=float(augment.blur_p),
                )
            )

        ops.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        if augment.erasing_p > 0:
            scale_min = float(augment.erasing_scale_min)
            scale_max = float(augment.erasing_scale_max)
            if scale_max < scale_min:
                scale_max = scale_min
            ratio_min = float(augment.erasing_ratio_min)
            ratio_max = float(augment.erasing_ratio_max)
            if ratio_max < ratio_min:
                ratio_max = ratio_min
            ops.append(
                transforms.RandomErasing(
                    p=float(augment.erasing_p),
                    scale=(scale_min, scale_max),
                    ratio=(ratio_min, ratio_max),
                    value=0,
                )
            )
        return transforms.Compose(ops)

    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def denormalize(img: torch.Tensor) -> torch.Tensor:
    """
    img: (3,H,W) normalized
    returns: (3,H,W) in [0,1]
    """
    mean = torch.tensor(IMAGENET_MEAN, device=img.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=img.device).view(3, 1, 1)
    x = img * std + mean
    return torch.clamp(x, 0.0, 1.0)
