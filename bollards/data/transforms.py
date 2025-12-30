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
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomResizedCrop(img_size, scale=(crop_min, crop_max)),
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
                interpolation=InterpolationMode.BILINEAR,
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
        return transforms.Compose(ops)

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
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
