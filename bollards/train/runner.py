from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from bollards.config import TrainConfig
from bollards.constants import BBOX_COLS, LABEL_COL, META_COLS, PATH_COL
from bollards.data.country_names import golden_country_to_code
from bollards.data.bboxes import compute_avg_bbox_wh
from bollards.data.datasets import BollardCropsDataset
from bollards.data.labels import load_id_to_country
from bollards.data.samplers import make_sampler
from bollards.data.transforms import build_transforms
from bollards.io.hf import hf_upload_run_artifacts
from bollards.models.bollard_net import BollardNet
from bollards.train.losses import FocalLoss
from bollards.train.loop import evaluate, train_one_epoch
from bollards.train.visuals import annotate_grid_images
from bollards.utils.seeding import make_torch_generator, seed_everything, seed_worker


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


def _compute_sqrt_inv_class_weights(labels: pd.Series, num_classes: int) -> list[float]:
    counts = labels.value_counts().reindex(range(num_classes), fill_value=0).sort_index()
    weights = []
    for c in counts.tolist():
        if c > 0:
            weights.append(1.0 / math.sqrt(float(c)))
        else:
            weights.append(0.0)
    nonzero = [w for w in weights if w > 0]
    if nonzero:
        mean = sum(nonzero) / len(nonzero)
        weights = [w / mean if w > 0 else 0.0 for w in weights]
    return weights


def _prepare_golden_df(
    golden_df: pd.DataFrame,
    country_to_id: Dict[str, int],
    *,
    avg_bbox_w: float,
    avg_bbox_h: float,
) -> pd.DataFrame:
    if avg_bbox_w is None or avg_bbox_h is None:
        raise ValueError("Golden dataset requires explicit avg_bbox_w/avg_bbox_h values.")

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
    meta_defaults = {
        "x_center": 0.5,
        "y_center": 0.5,
        "w": avg_bbox_w,
        "h": avg_bbox_h,
        "conf": 1.0,
    }
    for col in BBOX_COLS:
        df[col] = bbox_defaults[col]
    for col in META_COLS:
        df[col] = meta_defaults[col]

    df = df.drop(columns=["country_code"])
    return df


def _sample_diverse_rows(df: pd.DataFrame, n: int, label_col: str, seed: int = 42) -> pd.DataFrame:
    if n <= 0 or df.empty:
        return df.head(0).copy()
    if n >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if label_col not in df.columns:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)

    labels = df[label_col].dropna().unique().tolist()
    if not labels:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)

    if n <= len(labels):
        chosen = pd.Series(labels).sample(n=n, random_state=seed).tolist()
        sampled = (
            df[df[label_col].isin(chosen)]
            .groupby(label_col, group_keys=False)
            .sample(n=1, random_state=seed)
        )
        return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    base = df.groupby(label_col, group_keys=False).sample(n=1, random_state=seed)
    remaining = n - len(base)
    if remaining > 0:
        rest = df.drop(index=base.index)
        if not rest.empty:
            extra = rest.sample(n=min(remaining, len(rest)), random_state=seed)
            base = pd.concat([base, extra], ignore_index=True)

    return base.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _sanitize_config(cfg: TrainConfig) -> dict:
    data = asdict(cfg)
    hub_cfg = data.get("hub")
    if isinstance(hub_cfg, dict) and hub_cfg.get("token"):
        hub_cfg["token"] = "REDACTED"
    return data


def _format_template(template: Optional[str], **kwargs: object) -> Optional[str]:
    if not template:
        return template
    if "{" not in template:
        return template
    try:
        return template.format(**kwargs)
    except KeyError as exc:
        raise ValueError(f"Unknown placeholder in template: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"Invalid template format: {exc}") from exc


def _maybe_push_best_to_hf(
    cfg: TrainConfig,
    *,
    run_dir: str,
    run_name: str,
    best_metric_name: str,
    best_metric_value: float,
) -> None:
    if not cfg.hub.enabled:
        return

    repo_id = (cfg.hub.repo_id or "").strip()
    if not repo_id:
        raise ValueError("hub.enabled requires hub.repo_id")

    include = list(cfg.hub.upload_include or [])
    for required in ("best.pt", "config.json"):
        if required not in include:
            include.append(required)

    run_dir_path = Path(run_dir)
    best_path = run_dir_path / "best.pt"
    if not best_path.exists():
        print("[warn] best.pt not found; skipping Hugging Face upload.")
        return

    token = cfg.hub.token
    if not token and cfg.hub.token_env:
        token = os.getenv(cfg.hub.token_env)
        if not token:
            print(
                f"[warn] Hugging Face token env '{cfg.hub.token_env}' not set; "
                "falling back to cached credentials."
            )

    template_args = {
        "run_name": run_name,
        "best_metric": best_metric_name,
        "best_metric_value": best_metric_value,
    }
    path_in_repo = _format_template(cfg.hub.path_in_repo, **template_args)
    commit_message = _format_template(cfg.hub.commit_message, **template_args)

    uploaded = hf_upload_run_artifacts(
        repo_id=repo_id,
        run_dir=run_dir_path,
        allow_patterns=include,
        path_in_repo=path_in_repo,
        private=cfg.hub.private,
        commit_message=commit_message,
        token=token,
    )
    if uploaded:
        print(f"[info] uploaded {len(uploaded)} artifact(s) to huggingface.co/{repo_id}")
    else:
        print("[warn] no artifacts matched for Hugging Face upload.")


def run_training(cfg: TrainConfig) -> None:
    seed_everything(cfg.seed)

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
    use_class_weighting = bool(cfg.optim.class_weighting)
    use_sampler = bool(cfg.data.balanced_sampler)
    use_focal = bool(cfg.optim.focal_gamma > 0 or cfg.optim.focal_alpha is not None)
    enabled = {
        "class_weighting": use_class_weighting,
        "balanced_sampler": use_sampler,
        "focal_loss": use_focal,
    }
    if sum(enabled.values()) > 1:
        active = [k for k, v in enabled.items() if v]
        raise ValueError(
            "Only one of class_weighting, balanced_sampler, or focal_loss can be enabled at a time. "
            f"Active: {active}"
        )
    if cfg.data.max_train_samples > 0 and len(train_df) > cfg.data.max_train_samples:
        train_df = train_df.sample(n=cfg.data.max_train_samples, random_state=cfg.seed).reset_index(drop=True)
    if cfg.data.max_val_samples > 0 and len(val_df) > cfg.data.max_val_samples:
        val_df = val_df.sample(n=cfg.data.max_val_samples, random_state=cfg.seed).reset_index(drop=True)
    id_to_country, country_to_id = _build_country_mappings(id_to_country, train_df, val_df)
    avg_bbox_w, avg_bbox_h = compute_avg_bbox_wh(train_df, label="Train")

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

    train_tfm = build_transforms(train=True, img_size=cfg.data.img_size, augment=cfg.augment)
    val_tfm = build_transforms(train=False, img_size=cfg.data.img_size, augment=cfg.augment)

    train_ds = BollardCropsDataset(train_df, cfg.data.img_root, train_tfm, expand=cfg.data.expand)
    val_ds = BollardCropsDataset(val_df, cfg.data.img_root, val_tfm, expand=cfg.data.expand)

    sampler = None
    if cfg.data.balanced_sampler:
        sampler = make_sampler(
            train_df,
            generator=make_torch_generator(cfg.seed, "train_sampler"),
            alpha=cfg.data.sampler_alpha,
        )

    def _loader_kwargs(num_workers: int, generator: torch.Generator) -> dict:
        kwargs = {"num_workers": num_workers, "generator": generator}
        if num_workers > 0:
            kwargs["prefetch_factor"] = cfg.data.prefetch_factor
            kwargs["persistent_workers"] = cfg.data.persistent_workers
            kwargs["worker_init_fn"] = seed_worker
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
        **_loader_kwargs(train_workers, make_torch_generator(cfg.seed, "train_loader")),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        **_loader_kwargs(val_workers, make_torch_generator(cfg.seed, "val_loader")),
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
        golden_df = _prepare_golden_df(
            golden_df,
            country_to_id,
            avg_bbox_w=avg_bbox_w,
            avg_bbox_h=avg_bbox_h,
        )
        golden_root = cfg.data.golden_img_root or os.path.dirname(cfg.data.golden_csv) or "."
        golden_ds = BollardCropsDataset(golden_df, golden_root, val_tfm, expand=cfg.data.expand)
        golden_loader = DataLoader(
            golden_ds,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            **_loader_kwargs(golden_workers, make_torch_generator(cfg.seed, "golden_loader")),
        )
        if cfg.logging.log_images > 0:
            vis_count = min(cfg.logging.log_images, len(golden_df))
            golden_vis_df = _sample_diverse_rows(golden_df, vis_count, LABEL_COL, seed=cfg.seed)
            if not golden_vis_df.empty:
                golden_vis_ds = BollardCropsDataset(
                    golden_vis_df, golden_root, val_tfm, expand=cfg.data.expand
                )
                golden_vis_loader = DataLoader(
                    golden_vis_ds,
                    batch_size=min(cfg.data.batch_size, len(golden_vis_df)),
                    shuffle=False,
                    pin_memory=pin_memory,
                    **_loader_kwargs(golden_workers, make_torch_generator(cfg.seed, "golden_vis_loader")),
                )
                fixed_golden_batch = next(iter(golden_vis_loader))

    fixed_val_batch = next(iter(val_loader))

    model = BollardNet(cfg.model).to(device)
    print(f"[info] model config: {asdict(cfg.model)}")

    class_weights = None
    if cfg.optim.class_weighting:
        class_weights = _compute_sqrt_inv_class_weights(train_df[LABEL_COL], required_classes)
        print("[info] computed sqrt-inverse class weights from train labels.")

    focal_alpha = cfg.optim.focal_alpha
    if (cfg.optim.focal_gamma > 0 or focal_alpha is not None) and focal_alpha is None and class_weights is not None:
        focal_alpha = class_weights
        print("[info] computed focal_alpha from sqrt-inverse class weights.")

    if cfg.optim.focal_gamma > 0 or focal_alpha is not None:
        criterion = FocalLoss(
            gamma=cfg.optim.focal_gamma,
            alpha=focal_alpha,
            label_smoothing=cfg.optim.label_smoothing,
        )
    else:
        weight_tensor = None
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(
            label_smoothing=cfg.optim.label_smoothing,
            weight=weight_tensor,
            reduction="none",
        )

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
    config_payload = _sanitize_config(cfg)
    writer.add_text("run/config", json.dumps(config_payload, indent=2), global_step=0)
    writer.add_text("model/config", json.dumps(asdict(cfg.model), indent=2), global_step=0)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

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

    best_metric_name = str(cfg.logging.best_metric or "val_top1").strip()
    allowed_metrics = {
        "val_top1",
        "val_top5",
        "val_map",
        "golden_top1",
        "golden_top5",
        "golden_map",
    }
    if best_metric_name not in allowed_metrics:
        raise ValueError(
            f"logging.best_metric must be one of {sorted(allowed_metrics)} (got {best_metric_name})."
        )
    if best_metric_name.startswith("golden_") and golden_loader is None:
        print(
            f"[warn] best_metric={best_metric_name} but golden dataset is disabled; "
            "falling back to val_top1."
        )
        best_metric_name = "val_top1"

    best_metric_value = float("-inf")

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

        val_top1, val_top5, val_map = evaluate(model, val_loader, device, desc="val")
        golden_top1 = None
        golden_top5 = None
        golden_map = None
        if golden_loader is not None:
            golden_top1, golden_top5, golden_map = evaluate(
                model, golden_loader, device, desc="golden"
            )
        scheduler.step()

        lrs = [pg["lr"] for pg in optimizer.param_groups]
        if golden_top1 is None:
            print(
                f"[epoch {epoch:02d}] loss={train_loss:.4f} val_top1={val_top1:.4f} "
                f"val_top5={val_top5:.4f} val_map={val_map:.4f} lr={lrs}"
            )
        else:
            print(
                f"[epoch {epoch:02d}] loss={train_loss:.4f} val_top1={val_top1:.4f} "
                f"val_top5={val_top5:.4f} val_map={val_map:.4f} "
                f"golden_top1={golden_top1:.4f} golden_top5={golden_top5:.4f} "
                f"golden_map={golden_map:.4f} lr={lrs}"
            )

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/top1", val_top1, epoch)
        writer.add_scalar("val/top5", val_top5, epoch)
        writer.add_scalar("val/map", val_map, epoch)
        if golden_top1 is not None:
            writer.add_scalar("golden/top1", golden_top1, epoch)
            writer.add_scalar("golden/top5", golden_top5, epoch)
            writer.add_scalar("golden/map", golden_map, epoch)
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
        metrics = {
            "val_top1": val_top1,
            "val_top5": val_top5,
            "val_map": val_map,
            "golden_top1": golden_top1,
            "golden_top5": golden_top5,
            "golden_map": golden_map,
        }
        current_metric = metrics.get(best_metric_name)
        if current_metric is None:
            current_metric = val_top1
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "cfg": asdict(cfg.model),
                "val_top1": val_top1,
                "val_top5": val_top5,
                "val_map": val_map,
                "golden_top1": golden_top1,
                "golden_top5": golden_top5,
                "golden_map": golden_map,
                "best_metric_name": best_metric_name,
                "best_metric_value": current_metric,
            },
            last_path,
        )

        if current_metric > best_metric_value:
            best_metric_value = float(current_metric)
            best_path = os.path.join(run_dir, "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "cfg": asdict(cfg.model),
                    "val_top1": val_top1,
                    "val_top5": val_top5,
                    "val_map": val_map,
                    "golden_top1": golden_top1,
                    "golden_top5": golden_top5,
                    "golden_map": golden_map,
                    "best_metric_name": best_metric_name,
                    "best_metric_value": current_metric,
                },
                best_path,
            )
            print(
                f"[info] saved new best: {best_path} "
                f"({best_metric_name}={best_metric_value:.4f})"
            )

    writer.close()

    if cfg.hub.enabled:
        try:
            _maybe_push_best_to_hf(
                cfg,
                run_dir=run_dir,
                run_name=run_name,
                best_metric_name=best_metric_name,
                best_metric_value=best_metric_value,
            )
        except Exception as exc:
            if cfg.hub.fail_on_error:
                raise
            print(f"[warn] Hugging Face upload failed: {exc}")

    if cfg.analyze.enabled:
        from bollards.config import AnalyzeRunConfig, load_config, resolve_config_path
        from bollards.pipelines.analyze_run import run_analyze_run

        analyze_path = resolve_config_path(cfg.analyze.config_path, "analyze_run.json")
        analyze_cfg = load_config(analyze_path, AnalyzeRunConfig)
        analyze_cfg.data.training_run_dir = run_dir
        if not analyze_cfg.data.main_val_csv:
            analyze_cfg.data.main_val_csv = cfg.data.val_csv
        print(f"[info] running analyze_run from {analyze_path}")
        run_analyze_run(analyze_cfg)
