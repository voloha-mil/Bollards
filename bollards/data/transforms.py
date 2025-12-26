import torch
from torchvision import transforms

from bollards.constants import IMAGENET_MEAN, IMAGENET_STD


def build_transforms(train: bool, img_size: int) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
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
