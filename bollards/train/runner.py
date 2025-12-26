from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from bollards.config import TrainConfig
from bollards.constants import BBOX_COLS, LABEL_COL, META_COLS, PATH_COL
from bollards.data.country_names import golden_country_to_code
from bollards.data.datasets import BollardCropsDataset
from bollards.data.labels import load_id_to_country
from bollards.data.samplers import make_sampler
from bollards.data.transforms import build_transforms
from bollards.models.bollard_net import BollardNet
from bollards.train.losses import FocalLoss
from bollards.train.loop import evaluate, train_one_epoch
from bollards.train.visuals import annotate_grid_images


def _build_country_mappings(
    id_to_country: Optional[list[str]],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Tuple[Optional[list[str]], Optional[Dict[str, int]]]:
    if id_to_country:
        country_to_id = {name: idx for idx, name in enumerate(id_to_country) if name}
        return id_to_country, country_to_id

    combined = pd.concat([train_df, val_df], ignore_index=True)
    if "country" not in combined.columns or LABEL_COL not in combined.columns:
        return None, None

    pairs = combined[[LABEL_COL, "country"]].dropna().drop_duplicates()
    if pairs.empty:
        return None, None

    dup_country = pairs.groupby("country")[LABEL_COL].nunique()
    if (dup_country > 1).any():
        bad = dup_country[dup_country > 1].index.tolist()
        raise ValueError(f"Multiple ids for country names in training data: {bad}")

    dup_id = pairs.groupby(LABEL_COL)["country"].nunique()
    if (dup_id > 1).any():
        bad = dup_id[dup_id > 1].index.tolist()
        raise ValueError(f"Multiple country names for ids in training data: {bad}")

    country_to_id = {str(row["country"]): int(row[LABEL_COL]) for _, row in pairs.iterrows()}
    max_id = max(country_to_id.values())
    id_to_country = [""] * (max_id + 1)
    for name, idx in country_to_id.items():
        id_to_country[idx] = name

    return id_to_country, country_to_id


def _prepare_golden_df(golden_df: pd.DataFrame, country_to_id: Dict[str, int]) -> pd.DataFrame:
    required = [PATH_COL, "country"]
    missing = [c for c in required if c not in golden_df.columns]
    if missing:
        raise ValueError(f"Golden CSV missing required columns: {missing}")

    df = golden_df.copy()
    df["country_code"] = df["country"].apply(golden_country_to_code)
    if df["country_code"].isna().any():
        unknown = sorted(df.loc[df["country_code"].isna(), "country"].dropna().unique().tolist())
        print(f"[warn] golden CSV has unmapped countries; dropping: {unknown}")
        df = df.loc[df["country_code"].notna()].copy()

    df[LABEL_COL] = df["country_code"].map(country_to_id)
    if df[LABEL_COL].isna().any():
        unknown_codes = sorted(df.loc[df[LABEL_COL].isna(), "country_code"].dropna().unique().tolist())
        print(f"[warn] golden CSV has countries not in label map; dropping codes: {unknown_codes}")
        df = df.loc[df[LABEL_COL].notna()].copy()

    if df.empty:
        raise ValueError("Golden CSV has no rows after mapping countries to labels.")

    df[LABEL_COL] = df[LABEL_COL].astype(int)

    bbox_defaults = {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}
    meta_defaults = {"x_center": 0.5, "y_center": 0.5, "w": 1.0, "h": 1.0, "conf": 1.0}
    for col in BBOX_COLS:
        df[col] = bbox_defaults[col]
    for col in META_COLS:
        df[col] = meta_defaults[col]

    df = df.drop(columns=["country_code"])
    return df


def run_training(cfg: TrainConfig) -> None:
    if cfg.model.meta_dim != len(META_COLS):
        raise ValueError(f"model.meta_dim must match META_COLS ({len(META_COLS)})")

    os.makedirs(cfg.logging.out_dir, exist_ok=True)
    if cfg.logging.run_name:
        run_name = cfg.logging.run_name
    else:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.logging.out_dir, run_name)
    if os.path.exists(run_dir):
        suffix = 1
        while os.path.exists(f"{run_dir}_{suffix:02d}"):
            suffix += 1
        run_dir = f"{run_dir}_{suffix:02d}"
    os.makedirs(run_dir, exist_ok=True)

    tb_dir = cfg.logging.tb_dir or os.path.join(run_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    print(f"[info] run_dir: {run_dir}")

    if cfg.device and cfg.device != "auto":
        device = torch.device(cfg.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")

    id_to_country = load_id_to_country(cfg.data.country_map_json)

    train_df = pd.read_csv(cfg.data.train_csv)
    val_df = pd.read_csv(cfg.data.val_csv)
    if cfg.data.max_train_samples > 0 and len(train_df) > cfg.data.max_train_samples:
        train_df = train_df.sample(n=cfg.data.max_train_samples, random_state=42).reset_index(drop=True)
    if cfg.data.max_val_samples > 0 and len(val_df) > cfg.data.max_val_samples:
        val_df = val_df.sample(n=cfg.data.max_val_samples, random_state=42).reset_index(drop=True)
    id_to_country, country_to_id = _build_country_mappings(id_to_country, train_df, val_df)

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

    def _loader_kwargs(num_workers: int) -> dict:
        kwargs = {"num_workers": num_workers}
        if num_workers > 0:
            kwargs["prefetch_factor"] = cfg.data.prefetch_factor
            kwargs["persistent_workers"] = cfg.data.persistent_workers
        return kwargs

    pin_memory = device.type == "cuda"
    train_workers = cfg.data.num_workers
    val_workers = cfg.data.val_num_workers
    golden_workers = cfg.data.golden_num_workers

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        pin_memory=pin_memory,
        drop_last=True,
        **_loader_kwargs(train_workers),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        **_loader_kwargs(val_workers),
    )

    golden_loader = None
    fixed_golden_batch = None
    if cfg.data.golden_csv:
        if not country_to_id:
            raise ValueError(
                "golden_csv provided but no country mapping is available. "
                "Provide data.country_map_json or ensure train/val CSVs include country."
            )
        golden_df = pd.read_csv(cfg.data.golden_csv)
        golden_df = _prepare_golden_df(golden_df, country_to_id)
        golden_root = cfg.data.golden_img_root or os.path.dirname(cfg.data.golden_csv) or "."
        golden_ds = BollardCropsDataset(golden_df, golden_root, val_tfm, expand=cfg.data.expand)
        golden_loader = DataLoader(
            golden_ds,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            **_loader_kwargs(golden_workers),
        )
        fixed_golden_batch = next(iter(golden_loader))

    fixed_val_batch = next(iter(val_loader))

    model = BollardNet(cfg.model).to(device)
    print(f"[info] model config: {asdict(cfg.model)}")

    focal_alpha = cfg.optim.focal_alpha
    if (cfg.optim.focal_gamma > 0 or focal_alpha is not None) and focal_alpha is None:
        class_counts = train_df[LABEL_COL].value_counts().sort_index()
        inv = 1.0 / class_counts
        inv = inv / inv.mean()
        focal_alpha = inv.tolist()
        print("[info] computed focal_alpha from train class frequencies.")

    if cfg.optim.focal_gamma > 0 or focal_alpha is not None:
        criterion = FocalLoss(
            gamma=cfg.optim.focal_gamma,
            alpha=focal_alpha,
            label_smoothing=cfg.optim.label_smoothing,
        )
    else:
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
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    if cfg.data.country_map_json and os.path.exists(cfg.data.country_map_json):
        shutil.copyfile(cfg.data.country_map_json, os.path.join(run_dir, "country_map.json"))
        base_dir = os.path.dirname(cfg.data.country_map_json)
        for fname in ("country_list.json", "country_counts.csv"):
            src = os.path.join(base_dir, fname)
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(run_dir, fname))
        mapping_path = os.path.join(os.path.dirname(__file__), "..", "data", "country_mapping.json")
        mapping_path = os.path.normpath(mapping_path)
        if os.path.exists(mapping_path):
            shutil.copyfile(mapping_path, os.path.join(run_dir, "country_mapping.json"))

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

        val_top1, val_top5 = evaluate(model, val_loader, device, desc="val")
        golden_top1 = None
        golden_top5 = None
        if golden_loader is not None:
            golden_top1, golden_top5 = evaluate(model, golden_loader, device, desc="golden")
        scheduler.step()

        lrs = [pg["lr"] for pg in optimizer.param_groups]
        if golden_top1 is None:
            print(
                f"[epoch {epoch:02d}] loss={train_loss:.4f} val_top1={val_top1:.4f} "
                f"val_top5={val_top5:.4f} lr={lrs}"
            )
        else:
            print(
                f"[epoch {epoch:02d}] loss={train_loss:.4f} val_top1={val_top1:.4f} "
                f"val_top5={val_top5:.4f} golden_top1={golden_top1:.4f} "
                f"golden_top5={golden_top5:.4f} lr={lrs}"
            )

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/top1", val_top1, epoch)
        writer.add_scalar("val/top5", val_top5, epoch)
        if golden_top1 is not None:
            writer.add_scalar("golden/top1", golden_top1, epoch)
            writer.add_scalar("golden/top5", golden_top5, epoch)
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

            if golden_loader is not None and fixed_golden_batch is not None:
                images = fixed_golden_batch["image"].to(device, non_blocking=True)
                meta = fixed_golden_batch["meta"].to(device, non_blocking=True)
                labels = fixed_golden_batch["label"].to(device, non_blocking=True)

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

                writer.add_image("golden/examples_grid", grid, epoch)
                writer.add_text("golden/examples_table", f"<pre>{table}</pre>", epoch)

                writer.add_image(f"golden/examples_grid_epoch_{epoch:03d}", grid, 0)
                writer.add_text(f"golden/examples_table_epoch_{epoch:03d}", f"<pre>{table}</pre>", 0)

        last_path = os.path.join(run_dir, "last.pt")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "cfg": asdict(cfg.model),
                "val_top1": val_top1,
                "val_top5": val_top5,
                "golden_top1": golden_top1,
                "golden_top5": golden_top5,
            },
            last_path,
        )

        if val_top1 > best_top1:
            best_top1 = val_top1
            best_path = os.path.join(run_dir, "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "cfg": asdict(cfg.model),
                    "val_top1": val_top1,
                    "val_top5": val_top5,
                    "golden_top1": golden_top1,
                    "golden_top5": golden_top5,
                },
                best_path,
            )
            print(f"[info] saved new best: {best_path} (top1={best_top1:.4f})")

    writer.close()
