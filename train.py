import argparse
import os
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from classifier import BollardNet, ModelConfig


PATH_COL = "image_path"
LABEL_COL = "country_id"

META_COLS = ["x_center", "y_center", "w", "h", "conf"]


class BollardCropsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_root: str, tfm: transforms.Compose):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.tfm = tfm

        missing = [c for c in [PATH_COL, LABEL_COL, *META_COLS] if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        rel_path = str(row[PATH_COL])
        img_path = rel_path if os.path.isabs(rel_path) else os.path.join(self.img_root, rel_path)

        # Load RGB crop
        img = Image.open(img_path).convert("RGB")
        img = self.tfm(img)

        label = int(row[LABEL_COL])
        meta = row[META_COLS].astype(np.float32).to_numpy()

        # assumes [0,1] for coords + conf
        meta = np.clip(meta, 0.0, 1.0)

        return {
            "image": img,
            "meta": torch.from_numpy(meta),
            "label": torch.tensor(label, dtype=torch.long),
        }


def build_transforms(train: bool, img_size: int) -> transforms.Compose:
    # For geo tasks, horizontal flip can sometimes hurt (driving side cues),
    # so we keep it OFF by default.
    if train:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    correct1 = 0
    correct5 = 0
    total = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        meta = batch["meta"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = model(images, meta)
        total += labels.size(0)

        # top-1
        pred1 = logits.argmax(dim=1)
        correct1 += (pred1 == labels).sum().item()

        # top-5
        top5 = torch.topk(logits, k=min(5, logits.size(1)), dim=1).indices
        correct5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

    return correct1 / max(total, 1), correct5 / max(total, 1)


def balanced_country_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    # Balanced sampling by country_id
    counts = df[LABEL_COL].value_counts().to_dict()
    weights = df[LABEL_COL].map(lambda y: 1.0 / counts[int(y)]).astype(np.float32).to_numpy()
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    criterion: nn.Module,
    conf_weight_min: float,
) -> float:
    model.train()
    running_loss = 0.0
    n = 0

    use_amp = scaler is not None

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        meta = batch["meta"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(images, meta)
                # per-sample loss
                loss_vec = criterion(logits, labels)  # shape (B,)
        else:
            logits = model(images, meta)
            loss_vec = criterion(logits, labels)

        # confidence-weighted loss (uses detector conf as weight)
        # assumes conf is the last meta feature in META_COLS
        conf = meta[:, -1].detach()
        w = torch.clamp(conf, min=conf_weight_min, max=1.0)
        loss = (loss_vec * w).mean()

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += float(loss.item()) * labels.size(0)
        n += labels.size(0)

    return running_loss / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--val_csv", type=str, required=True)
    ap.add_argument("--img_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="runs/bollard_country")

    ap.add_argument("--num_classes", type=int, required=True)

    ap.add_argument("--backbone", type=str, default="efficientnet_b1")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--freeze_epochs", type=int, default=1)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--backbone_lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--conf_weight_min", type=float, default=0.2)

    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--balanced_sampler", action="store_true")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")

    # Load data
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    train_tfm = build_transforms(train=True, img_size=args.img_size)
    val_tfm = build_transforms(train=False, img_size=args.img_size)

    train_ds = BollardCropsDataset(train_df, args.img_root, train_tfm)
    val_ds = BollardCropsDataset(val_df, args.img_root, val_tfm)

    sampler = balanced_country_sampler(train_df) if args.balanced_sampler else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Model
    cfg = ModelConfig(
        backbone_name=args.backbone,
        pretrained=True,
        num_classes=args.num_classes,
        meta_dim=len(META_COLS),
    )
    model = BollardNet(cfg).to(device)
    print(f"[info] model config: {asdict(cfg)}")

    # Loss: per-sample CE so we can apply confidence weights
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction="none")

    # Optimizer with 2 parameter groups (slower LR for backbone)
    def param_groups(m: BollardNet):
        backbone_params = [p for p in m.backbone.parameters() if p.requires_grad]
        head_params = [p for n, p in m.named_parameters() if n.startswith("meta_mlp") or n.startswith("head")]
        return [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params, "lr": args.lr},
        ]

    # Stage 1: freeze backbone (optional)
    if args.freeze_epochs > 0:
        model.freeze_backbone()
    optimizer = torch.optim.AdamW(param_groups(model), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_top1 = 0.0
    for epoch in range(1, args.epochs + 1):
        # unfreeze after freeze_epochs
        if epoch == args.freeze_epochs + 1:
            print("[info] unfreezing backbone for fine-tuning")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(param_groups(model), weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - epoch + 1)

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            criterion=criterion,
            conf_weight_min=args.conf_weight_min,
        )

        top1, top5 = evaluate(model, val_loader, device)
        scheduler.step()

        lrs = [pg["lr"] for pg in optimizer.param_groups]
        print(f"[epoch {epoch:02d}] loss={train_loss:.4f} val_top1={top1:.4f} val_top5={top5:.4f} lr={lrs}")

        # Save last
        last_path = os.path.join(args.out_dir, "last.pt")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "cfg": asdict(cfg),
                "val_top1": top1,
                "val_top5": top5,
            },
            last_path,
        )

        # Save best
        if top1 > best_top1:
            best_top1 = top1
            best_path = os.path.join(args.out_dir, "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "cfg": asdict(cfg),
                    "val_top1": top1,
                    "val_top5": top5,
                },
                best_path,
            )
            print(f"[info] saved new best: {best_path} (top1={best_top1:.4f})")


if __name__ == "__main__":
    main()
