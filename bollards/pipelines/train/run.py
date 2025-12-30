from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from bollards.pipelines.train.config import TrainConfig
from bollards.constants import LABEL_COL, META_COLS
from bollards.data.bboxes import compute_avg_bbox_wh
from bollards.data.datasets import BollardCropsDataset
from bollards.data.labels import load_id_to_country
from bollards.data.samplers import make_sampler
from bollards.data.transforms import build_transforms
from bollards.pipelines.train.data import (
    _build_country_mappings,
    _compute_sqrt_inv_class_weights,
    _prepare_golden_df,
    _sample_diverse_rows,
)
from bollards.pipelines.train.hub import _maybe_push_best_to_hf, _sanitize_config
from bollards.pipelines.train.loop import evaluate, train_one_epoch
from bollards.pipelines.train.losses import FocalLoss
from bollards.models.classifier import BollardNet
from bollards.utils.visuals.annotate import annotate_grid_images
from bollards.utils.seeding import make_torch_generator, seed_everything, seed_worker


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

    def build_scheduler(optimizer: torch.optim.Optimizer, remaining_epochs: int):
        name = str(cfg.schedule.lr_scheduler or "cosine").strip().lower()
        if name in ("cosine", "cosineannealing", "cosine_annealing"):
            t_max = cfg.schedule.cosine.t_max
            if t_max is None:
                t_max = max(1, remaining_epochs)
            if t_max <= 0:
                raise ValueError("schedule.cosine.t_max must be >= 1")
            return (
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=t_max,
                    eta_min=cfg.schedule.cosine.eta_min,
                ),
                False,
            )
        if name in ("reduce_on_plateau", "plateau", "reduce"):
            sched_cfg = cfg.schedule.reduce_on_plateau
            return (
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode=sched_cfg.mode,
                    factor=sched_cfg.factor,
                    patience=sched_cfg.patience,
                    threshold=sched_cfg.threshold,
                    threshold_mode=sched_cfg.threshold_mode,
                    cooldown=sched_cfg.cooldown,
                    min_lr=sched_cfg.min_lr,
                    eps=sched_cfg.eps,
                ),
                True,
            )
        raise ValueError(f"Unknown lr_scheduler: {cfg.schedule.lr_scheduler}")

    if cfg.schedule.freeze_epochs > 0:
        model.freeze_backbone()

    optimizer = torch.optim.AdamW(param_groups(model), weight_decay=cfg.optim.weight_decay)
    scheduler, scheduler_requires_metric = build_scheduler(optimizer, cfg.schedule.epochs)
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

    scheduler_monitor = None
    if scheduler_requires_metric:
        scheduler_monitor = str(cfg.schedule.reduce_on_plateau.monitor).strip()
        if scheduler_monitor not in allowed_metrics:
            raise ValueError(
                "schedule.reduce_on_plateau.monitor must be one of "
                f"{sorted(allowed_metrics)} (got {scheduler_monitor})."
            )
        if scheduler_monitor.startswith("golden_") and golden_loader is None:
            print(
                f"[warn] schedule.reduce_on_plateau.monitor={scheduler_monitor} but golden dataset is disabled; "
                "falling back to val_top1."
            )
            scheduler_monitor = "val_top1"

    best_metric_value = float("-inf")

    for epoch in range(1, cfg.schedule.epochs + 1):
        if epoch == cfg.schedule.freeze_epochs + 1:
            print("[info] unfreezing backbone for fine-tuning")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(param_groups(model), weight_decay=cfg.optim.weight_decay)
            scheduler, scheduler_requires_metric = build_scheduler(
                optimizer, cfg.schedule.epochs - epoch + 1
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
        metrics = {
            "val_top1": val_top1,
            "val_top5": val_top5,
            "val_map": val_map,
            "golden_top1": golden_top1,
            "golden_top5": golden_top5,
            "golden_map": golden_map,
        }
        if scheduler_requires_metric:
            scheduler_metric = metrics.get(scheduler_monitor)
            if scheduler_metric is None:
                raise ValueError(
                    f"schedule.reduce_on_plateau.monitor={scheduler_monitor} is unavailable for this run."
                )
            scheduler.step(float(scheduler_metric))
        else:
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
        from bollards.pipelines.analyze.config import AnalyzeRunConfig
        from bollards.utils.config import load_config, resolve_config_path
        from bollards.pipelines.analyze.run import run_analyze_run

        analyze_path = resolve_config_path(cfg.analyze.config_path, "analyze_run.json")
        analyze_cfg = load_config(analyze_path, AnalyzeRunConfig)
        analyze_cfg.data.training_run_dir = run_dir
        if not analyze_cfg.data.main_val_csv:
            analyze_cfg.data.main_val_csv = cfg.data.val_csv
        print(f"[info] running analyze_run from {analyze_path}")
        run_analyze_run(analyze_cfg)
