# train.py
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid

from classifier import BollardNet, ModelConfig

from tqdm import tqdm


# CSV schema produced by your prep script
PATH_COL = "image_path"
LABEL_COL = "country_id"
COUNTRY_STR_COL = "country"  # optional, for nicer logs
META_COLS = ["x_center", "y_center", "w", "h", "conf"]
BBOX_COLS = ["x1", "y1", "x2", "y2"]  # normalized in [0,1]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BollardCropsDataset(Dataset):
    """
    Loads original image, crops expanded bbox, applies transforms.
    """

    def __init__(self, df: pd.DataFrame, img_root: str, tfm: transforms.Compose, expand: float = 2.0):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.tfm = tfm
        self.expand = expand

        required = [PATH_COL, LABEL_COL, *BBOX_COLS, *META_COLS]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        rel_path = str(row[PATH_COL])
        img_path = rel_path if os.path.isabs(rel_path) else os.path.join(self.img_root, rel_path)

        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        x1, y1, x2, y2 = [float(row[c]) for c in BBOX_COLS]

        # Expand bbox around center in normalized coords
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bw = (x2 - x1) * self.expand
        bh = (y2 - y1) * self.expand

        ex1 = max(0.0, cx - 0.5 * bw)
        ey1 = max(0.0, cy - 0.5 * bh)
        ex2 = min(1.0, cx + 0.5 * bw)
        ey2 = min(1.0, cy + 0.5 * bh)

        px1 = int(round(ex1 * W))
        py1 = int(round(ey1 * H))
        px2 = int(round(ex2 * W))
        py2 = int(round(ey2 * H))
        if px2 <= px1:
            px2 = min(W, px1 + 1)
        if py2 <= py1:
            py2 = min(H, py1 + 1)

        crop = img.crop((px1, py1, px2, py2))
        crop = self.tfm(crop)

        label = int(row[LABEL_COL])
        meta = row[META_COLS].astype("float32").to_numpy()
        meta = np.clip(meta, 0.0, 1.0)

        return {
            "image": crop,
            "meta": torch.from_numpy(meta),
            "label": torch.tensor(label, dtype=torch.long),
        }


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
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def make_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    counts = df[LABEL_COL].value_counts().to_dict()
    weights = df[LABEL_COL].map(lambda y: 1.0 / counts[int(y)]).astype(np.float32).to_numpy()
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def denormalize(img: torch.Tensor) -> torch.Tensor:
    """
    img: (3,H,W) normalized
    returns: (3,H,W) in [0,1]
    """
    mean = torch.tensor(IMAGENET_MEAN, device=img.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=img.device).view(3, 1, 1)
    x = img * std + mean
    return torch.clamp(x, 0.0, 1.0)


def _load_font(font_size: int) -> ImageFont.ImageFont:
    """
    Load a readable TrueType font if available; fall back to PIL default.
    Works well on Ubuntu (DejaVu) and macOS (Arial).
    """
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for p in candidates:
        try:
            if os.path.exists(p):
                return ImageFont.truetype(p, size=font_size)
        except Exception:
            pass
    return ImageFont.load_default()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    correct1 = 0
    correct5 = 0
    total = 0

    pbar = tqdm(loader, desc="val", leave=False, dynamic_ncols=True)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        meta = batch["meta"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = model(images, meta)
        total += labels.size(0)

        pred1 = logits.argmax(dim=1)
        correct1 += (pred1 == labels).sum().item()

        top5 = torch.topk(logits, k=min(5, logits.size(1)), dim=1).indices
        correct5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

        pbar.set_postfix_str(f"top1={correct1/max(total,1):.3f} top5={correct5/max(total,1):.3f}")

    return correct1 / max(total, 1), correct5 / max(total, 1)


def annotate_grid_images(
    images_norm: torch.Tensor,
    y_true: torch.Tensor,
    logits: torch.Tensor,
    id_to_country: Optional[List[str]],
    max_items: int = 16,
    topk: int = 3,
    font_size: int = 18,
) -> Tuple[torch.Tensor, str]:
    """
    Creates an annotated image grid and a text table summary.

    Adds:
      - T: true label name
      - P: predicted label name
      - p(T): probability assigned to the true class
      - top-k predictions with probabilities
    """
    B = images_norm.size(0)
    n = min(B, max_items)

    probs = torch.softmax(logits[:n], dim=1)
    topv, topi = torch.topk(probs, k=min(topk, probs.size(1)), dim=1)

    font = _load_font(font_size)
    annotated = []
    lines = ["idx\ttrue\tpred\tp(true)\t(topk)"]

    for i in range(n):
        img = denormalize(images_norm[i]).cpu()
        pil = transforms.ToPILImage()(img)

        yt = int(y_true[i].item())
        yp = int(topi[i, 0].item())

        true_name = id_to_country[yt] if id_to_country and yt < len(id_to_country) else str(yt)
        pred_name = id_to_country[yp] if id_to_country and yp < len(id_to_country) else str(yp)
        p_true = float(probs[i, yt].item()) if yt < probs.size(1) else 0.0

        topk_str = []
        for k in range(topi.size(1)):
            cid = int(topi[i, k].item())
            name = id_to_country[cid] if id_to_country and cid < len(id_to_country) else str(cid)
            topk_str.append(f"{name}:{float(topv[i, k].item()):.2f}")

        draw = ImageDraw.Draw(pil)
        text = f"T:{true_name}  P:{pred_name}  p(T)={p_true:.2f}\n" + "  ".join(topk_str)

        bg_h = int(font_size * 2.8)
        draw.rectangle((0, 0, pil.size[0], bg_h), fill=(0, 0, 0))
        draw.multiline_text((6, 4), text, fill=(255, 255, 255), font=font, spacing=2)

        annotated.append(transforms.ToTensor()(pil))
        lines.append(f"{i}\t{true_name}\t{pred_name}\t{p_true:.3f}\t{' | '.join(topk_str)}")

    grid = make_grid(torch.stack(annotated, dim=0), nrow=4, padding=2)
    table = "\n".join(lines)
    return grid, table


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
    running = 0.0
    n = 0
    use_amp = scaler is not None

    pbar = tqdm(loader, desc="train", leave=False, dynamic_ncols=True)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        meta = batch["meta"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(images, meta)
                loss_vec = criterion(logits, labels)  # (B,)
        else:
            logits = model(images, meta)
            loss_vec = criterion(logits, labels)

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

        running += float(loss.item()) * labels.size(0)
        n += labels.size(0)
        pbar.set_postfix_str(f"loss={running/max(n,1):.4f}")

    return running / max(n, 1)


def load_id_to_country(country_map_json: Optional[str]) -> Optional[List[str]]:
    if not country_map_json:
        return None
    with open(country_map_json, "r", encoding="utf-8") as f:
        m = json.load(f)  # country(str)->id(int)
    inv = [""] * (max(m.values()) + 1)
    for k, v in m.items():
        inv[int(v)] = str(k)
    return inv


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--val_csv", type=str, required=True)
    ap.add_argument("--img_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="runs/bollard_country")

    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--country_map_json", type=str, default=None, help="path to country_map.json for nicer logs")

    ap.add_argument("--backbone", type=str, default="efficientnet_b1")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--freeze_epochs", type=int, default=1)
    ap.add_argument("--expand", type=float, default=2.0, help="bbox expansion factor")

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--backbone_lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--conf_weight_min", type=float, default=0.2)

    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--balanced_sampler", action="store_true")

    # TensorBoard
    ap.add_argument("--tb_dir", type=str, default=None, help="defaults to <out_dir>/tb")
    ap.add_argument("--log_images", type=int, default=16, help="num val crops to visualize per epoch")
    ap.add_argument("--log_image_every", type=int, default=1, help="log images every N epochs")
    ap.add_argument("--tb_font_size", type=int, default=18, help="font size for image overlays")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tb_dir = args.tb_dir or os.path.join(args.out_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")

    id_to_country = load_id_to_country(args.country_map_json)

    # Data
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    train_tfm = build_transforms(train=True, img_size=args.img_size)
    val_tfm = build_transforms(train=False, img_size=args.img_size)

    train_ds = BollardCropsDataset(train_df, args.img_root, train_tfm, expand=args.expand)
    val_ds = BollardCropsDataset(val_df, args.img_root, val_tfm, expand=args.expand)

    sampler = make_sampler(train_df) if args.balanced_sampler else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Fixed val batch: same examples each epoch, easy comparison
    fixed_val_batch = next(iter(val_loader))

    # Model
    cfg = ModelConfig(
        backbone_name=args.backbone,
        pretrained=True,
        num_classes=args.num_classes,
        meta_dim=len(META_COLS),
    )
    model = BollardNet(cfg).to(device)
    print(f"[info] model config: {asdict(cfg)}")

    # Loss: per-sample CE so we can weight by conf
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

    # TensorBoard writer
    writer = SummaryWriter(log_dir=tb_dir)
    writer.add_text("run/config", json.dumps(vars(args), indent=2), global_step=0)
    writer.add_text("model/config", json.dumps(asdict(cfg), indent=2), global_step=0)

    best_top1 = 0.0

    for epoch in range(1, args.epochs + 1):
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

        val_top1, val_top5 = evaluate(model, val_loader, device)
        scheduler.step()

        lrs = [pg["lr"] for pg in optimizer.param_groups]
        print(f"[epoch {epoch:02d}] loss={train_loss:.4f} val_top1={val_top1:.4f} val_top5={val_top5:.4f} lr={lrs}")

        # ---- TensorBoard scalars ----
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/top1", val_top1, epoch)
        writer.add_scalar("val/top5", val_top5, epoch)
        writer.add_scalar("lr/backbone", lrs[0], epoch)
        writer.add_scalar("lr/head", lrs[1], epoch)

        # ---- TensorBoard visuals: fixed val crops with predictions (per epoch) ----
        if (epoch % args.log_image_every) == 0 and args.log_images > 0:
            model.eval()
            images = fixed_val_batch["image"].to(device, non_blocking=True)
            meta = fixed_val_batch["meta"].to(device, non_blocking=True)
            labels = fixed_val_batch["label"].to(device, non_blocking=True)

            with torch.no_grad():
                logits = model(images, meta)

            grid, table = annotate_grid_images(
                images_norm=images,
                y_true=labels,
                logits=logits,
                id_to_country=id_to_country,
                max_items=args.log_images,
                topk=3,
                font_size=args.tb_font_size,
            )

            # Step-based browsing (TensorBoard step selector)
            writer.add_image("val/examples_grid", grid, epoch)
            writer.add_text("val/examples_table", f"<pre>{table}</pre>", epoch)

            # Epoch-specific tags (each epoch is a separate entry you can click)
            writer.add_image(f"val/examples_grid_epoch_{epoch:03d}", grid, 0)
            writer.add_text(f"val/examples_table_epoch_{epoch:03d}", f"<pre>{table}</pre>", 0)

        # ---- Save checkpoints ----
        last_path = os.path.join(args.out_dir, "last.pt")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "cfg": asdict(cfg),
                "val_top1": val_top1,
                "val_top5": val_top5,
            },
            last_path,
        )

        if val_top1 > best_top1:
            best_top1 = val_top1
            best_path = os.path.join(args.out_dir, "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "cfg": asdict(cfg),
                    "val_top1": val_top1,
                    "val_top5": val_top5,
                },
                best_path,
            )
            print(f"[info] saved new best: {best_path} (top1={best_top1:.4f})")

    writer.close()


if __name__ == "__main__":
    main()
