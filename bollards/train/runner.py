from __future__ import annotations

import json
import os
from dataclasses import asdict

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from bollards.config import TrainConfig
from bollards.constants import LABEL_COL, META_COLS
from bollards.data.datasets import BollardCropsDataset
from bollards.data.labels import load_id_to_country
from bollards.data.samplers import make_sampler
from bollards.data.transforms import build_transforms
from bollards.models.bollard_net import BollardNet
from bollards.train.loop import evaluate, train_one_epoch
from bollards.train.visuals import annotate_grid_images


def run_training(cfg: TrainConfig) -> None:
    if cfg.model.meta_dim != len(META_COLS):
        raise ValueError(f"model.meta_dim must match META_COLS ({len(META_COLS)})")

    os.makedirs(cfg.logging.out_dir, exist_ok=True)
    tb_dir = cfg.logging.tb_dir or os.path.join(cfg.logging.out_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)

    if cfg.device and cfg.device != "auto":
        device = torch.device(cfg.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")

    id_to_country = load_id_to_country(cfg.data.country_map_json)

    train_df = pd.read_csv(cfg.data.train_csv)
    val_df = pd.read_csv(cfg.data.val_csv)

    data_max_label = int(max(train_df[LABEL_COL].max(), val_df[LABEL_COL].max()))
    if id_to_country is not None:
        required_classes = len(id_to_country)
        if data_max_label >= required_classes:
            raise ValueError(
                f"Label {data_max_label} is out of bounds for country_map_json "
                f"(len={required_classes})."
            )
    else:
        required_classes = data_max_label + 1

    if cfg.model.num_classes != required_classes:
        print(
            f"[info] overriding num_classes={cfg.model.num_classes} "
            f"with required_classes={required_classes}"
        )
        cfg.model.num_classes = required_classes

    train_tfm = build_transforms(train=True, img_size=cfg.data.img_size)
    val_tfm = build_transforms(train=False, img_size=cfg.data.img_size)

    train_ds = BollardCropsDataset(train_df, cfg.data.img_root, train_tfm, expand=cfg.data.expand)
    val_ds = BollardCropsDataset(val_df, cfg.data.img_root, val_tfm, expand=cfg.data.expand)

    sampler = make_sampler(train_df) if cfg.data.balanced_sampler else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    fixed_val_batch = next(iter(val_loader))

    model = BollardNet(cfg.model).to(device)
    print(f"[info] model config: {asdict(cfg.model)}")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.optim.label_smoothing, reduction="none")

    def param_groups(m: BollardNet):
        backbone_params = [p for p in m.backbone.parameters() if p.requires_grad]
        head_params = [p for n, p in m.named_parameters() if n.startswith("meta_mlp") or n.startswith("head")]
        return [
            {"params": backbone_params, "lr": cfg.optim.backbone_lr},
            {"params": head_params, "lr": cfg.optim.lr},
        ]

    if cfg.schedule.freeze_epochs > 0:
        model.freeze_backbone()

    optimizer = torch.optim.AdamW(param_groups(model), weight_decay=cfg.optim.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.schedule.epochs)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    writer = SummaryWriter(log_dir=tb_dir)
    writer.add_text("run/config", json.dumps(asdict(cfg), indent=2), global_step=0)
    writer.add_text("model/config", json.dumps(asdict(cfg.model), indent=2), global_step=0)

    best_top1 = 0.0

    for epoch in range(1, cfg.schedule.epochs + 1):
        if epoch == cfg.schedule.freeze_epochs + 1:
            print("[info] unfreezing backbone for fine-tuning")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(param_groups(model), weight_decay=cfg.optim.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.schedule.epochs - epoch + 1
            )

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            criterion=criterion,
            conf_weight_min=cfg.optim.conf_weight_min,
        )

        val_top1, val_top5 = evaluate(model, val_loader, device)
        scheduler.step()

        lrs = [pg["lr"] for pg in optimizer.param_groups]
        print(
            f"[epoch {epoch:02d}] loss={train_loss:.4f} val_top1={val_top1:.4f} val_top5={val_top5:.4f} lr={lrs}"
        )

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/top1", val_top1, epoch)
        writer.add_scalar("val/top5", val_top5, epoch)
        writer.add_scalar("lr/backbone", lrs[0], epoch)
        writer.add_scalar("lr/head", lrs[1], epoch)

        if (epoch % cfg.logging.log_image_every) == 0 and cfg.logging.log_images > 0:
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
                max_items=cfg.logging.log_images,
                topk=3,
                font_size=cfg.logging.tb_font_size,
            )

            writer.add_image("val/examples_grid", grid, epoch)
            writer.add_text("val/examples_table", f"<pre>{table}</pre>", epoch)

            writer.add_image(f"val/examples_grid_epoch_{epoch:03d}", grid, 0)
            writer.add_text(f"val/examples_table_epoch_{epoch:03d}", f"<pre>{table}</pre>", 0)

        last_path = os.path.join(cfg.logging.out_dir, "last.pt")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "cfg": asdict(cfg.model),
                "val_top1": val_top1,
                "val_top5": val_top5,
            },
            last_path,
        )

        if val_top1 > best_top1:
            best_top1 = val_top1
            best_path = os.path.join(cfg.logging.out_dir, "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "cfg": asdict(cfg.model),
                    "val_top1": val_top1,
                    "val_top5": val_top5,
                },
                best_path,
            )
            print(f"[info] saved new best: {best_path} (top1={best_top1:.4f})")

    writer.close()
