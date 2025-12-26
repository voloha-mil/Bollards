from __future__ import annotations

import json
import logging
import os
import queue
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from bollards.config import LiveScreenConfig
from bollards.data.labels import load_id_to_country
from bollards.data.transforms import build_transforms
from bollards.detect.yolo import run_inference
from bollards.io.fs import ensure_dir
from bollards.live.visuals import GridItem, GridViewer, render_grid
from bollards.data.bboxes import (
    bbox_xyxy_norm_to_center,
    crop_image_from_norm_bbox,
    normalize_bbox_xyxy_px,
)
from bollards.pipelines.common import load_classifier, load_detector, resolve_device, setup_logger


def _capture_screen(sct, cfg: LiveScreenConfig) -> Image.Image:
    if cfg.capture.region:
        monitor = cfg.capture.region
    else:
        monitors = sct.monitors
        if cfg.capture.monitor_index < 0 or cfg.capture.monitor_index >= len(monitors):
            raise SystemExit(f"monitor_index out of range: {cfg.capture.monitor_index}")
        monitor = monitors[cfg.capture.monitor_index]
    shot = sct.grab(monitor)
    return Image.frombytes("RGB", shot.size, shot.rgb)


def _filter_detections(cfg: LiveScreenConfig, boxes_xyxy, confs, clss) -> list[dict[str, float]]:
    allow_set = set(cfg.filters.cls_allow) if cfg.filters.cls_allow is not None else None
    filtered = []
    for i in range(len(confs)):
        conf = float(confs[i])
        cls = float(clss[i])
        x1, y1, x2, y2 = [float(v) for v in boxes_xyxy[i]]

        if conf < cfg.filters.min_conf:
            continue
        if allow_set is not None and cls not in allow_set:
            continue
        if abs(x2 - x1) < cfg.filters.min_box_w_px:
            continue
        if abs(y2 - y1) < cfg.filters.min_box_h_px:
            continue

        filtered.append({
            "conf": conf,
            "cls": cls,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        })

    filtered.sort(key=lambda d: d["conf"], reverse=True)
    max_keep = max(1, int(cfg.filters.max_boxes_per_image))
    return filtered[:max_keep]


def _crop_and_meta(img: Image.Image, det: dict[str, float], expand: float) -> tuple[Image.Image, list[float], list[float]]:
    w, h = img.size
    x1n, y1n, x2n, y2n = normalize_bbox_xyxy_px(det["x1"], det["y1"], det["x2"], det["y2"], w, h)

    xc, yc, bw, bh = bbox_xyxy_norm_to_center(x1n, y1n, x2n, y2n)
    conf = float(max(0.0, min(1.0, det["conf"])))
    crop = crop_image_from_norm_bbox(img, x1n, y1n, x2n, y2n, expand)
    meta = [xc, yc, bw, bh, conf]
    bbox_norm = [x1n, y1n, x2n, y2n]
    return crop, meta, bbox_norm


def _setup_hotkey_listener(hotkey: str, trigger_queue: queue.Queue, logger: logging.Logger):
    try:
        from pynput import keyboard
    except Exception as exc:
        logger.warning("pynput not available for hotkeys (%s)", exc)
        return None

    def on_activate():
        trigger_queue.put(time.time())

    listener = keyboard.GlobalHotKeys({hotkey: on_activate})
    listener.start()
    return listener


def run_live_screen(cfg: LiveScreenConfig) -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    out_dir = Path(cfg.output.out_dir)
    ensure_dir(out_dir)

    run_name = cfg.output.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = out_dir / run_name
    if run_dir.exists():
        suffix = 1
        while (out_dir / f"{run_name}_{suffix:02d}").exists():
            suffix += 1
        run_dir = out_dir / f"{run_name}_{suffix:02d}"
    ensure_dir(run_dir)

    screenshots_dir = run_dir / "screens"
    crops_dir = run_dir / "crops"
    ensure_dir(screenshots_dir)
    ensure_dir(crops_dir)

    log_path = run_dir / "session.log"
    logger = setup_logger("live_screen", log_path)

    config_path = run_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    device = resolve_device(cfg.device)
    logger.info("Using device: %s", device)

    detector = load_detector(
        weights_path=cfg.detector.weights_path,
        hf_repo=cfg.detector.hf_repo,
        hf_filename=cfg.detector.hf_filename,
        hf_cache=cfg.detector.hf_cache,
        logger=logger,
    )
    classifier, model_cfg = load_classifier(
        checkpoint_path=cfg.classifier.checkpoint_path,
        device=device,
        logger=logger,
    )
    id_to_country = load_id_to_country(cfg.classifier.country_map_json)

    transform = build_transforms(train=False, img_size=cfg.classifier.img_size)

    grid_items: list[GridItem] = []
    captures_total = 0
    captures_with_boxes = 0
    accepted_total = 0
    sum_top1_conf = 0.0
    sum_probs = np.zeros(model_cfg.num_classes, dtype=np.float64)

    jsonl_path = run_dir / "session.jsonl"
    jsonl_f = jsonl_path.open("a", encoding="utf-8")

    viewer: Optional[GridViewer] = None
    if cfg.output.viewer_enabled:
        try:
            if os.environ.get("DISPLAY") or os.name == "nt":
                viewer = GridViewer()
            else:
                logger.info("Viewer disabled (no DISPLAY).")
        except Exception as exc:
            logger.info("Viewer disabled (%s)", exc)
            viewer = None

    def id_to_name(idx: int) -> str:
        if id_to_country and idx < len(id_to_country):
            return id_to_country[idx]
        return str(idx)

    def update_grid() -> None:
        grid_img = render_grid(
            grid_items,
            cols=cfg.output.grid_cols,
            thumb_size=cfg.output.grid_thumb_size,
            max_items=cfg.output.grid_max_items,
        )
        if grid_img is None:
            return
        if cfg.output.save_grid:
            grid_img.save(run_dir / "grid.jpg")
        if viewer is not None:
            viewer.update(grid_img)

    def log_summary() -> dict:
        avg_conf = (sum_top1_conf / accepted_total) if accepted_total else 0.0
        top3 = []
        if accepted_total:
            avg_probs = sum_probs / float(accepted_total)
            top_idx = np.argsort(avg_probs)[::-1][:3]
            for idx in top_idx:
                top3.append({
                    "country_id": int(idx),
                    "country": id_to_name(int(idx)),
                    "avg_prob": float(avg_probs[idx]),
                })
        summary = {
            "captures": captures_total,
            "captures_with_boxes": captures_with_boxes,
            "accepted": accepted_total,
            "avg_confidence": avg_conf,
            "top3_avg_prob": top3,
        }
        return summary

    def handle_capture(screen: Image.Image, screen_path: Path) -> None:
        nonlocal captures_total, captures_with_boxes, accepted_total, sum_top1_conf, sum_probs
        captures_total += 1
        results = run_inference(
            model=detector,
            image_path=screen_path,
            imgsz=cfg.detector.imgsz,
            conf=cfg.detector.conf,
            device=str(device),
        )
        if not results:
            logger.info("No detection results")
            return

        r0 = results[0]
        boxes = r0.boxes
        if boxes is None or len(boxes) == 0:
            logger.info("No boxes detected")
            return

        boxes_xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy()
        filtered = _filter_detections(cfg, boxes_xyxy, confs, clss)
        if not filtered:
            logger.info("No boxes after filters")
            return

        captures_with_boxes += 1

        crops = []
        metas = []
        infos = []
        for det in filtered:
            crop, meta, bbox_norm = _crop_and_meta(screen, det, cfg.classifier.expand)
            crops.append(transform(crop))
            metas.append(meta)
            infos.append((det, bbox_norm))

        images_t = torch.stack(crops, dim=0).to(device, non_blocking=True)
        meta_t = torch.tensor(metas, dtype=torch.float32, device=device)

        with torch.no_grad():
            logits = classifier(images_t, meta_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        for i, (det, bbox_norm) in enumerate(infos):
            prob_vec = probs[i]
            topk = max(1, min(cfg.classifier.topk, prob_vec.shape[0]))
            top_idx = np.argsort(prob_vec)[::-1][:topk]

            top1_id = int(top_idx[0])
            top1_prob = float(prob_vec[top1_id])
            topk_list = [
                {
                    "country_id": int(idx),
                    "country": id_to_name(int(idx)),
                    "prob": float(prob_vec[idx]),
                }
                for idx in top_idx
            ]

            # if top1_prob < cfg.classifier.min_class_conf:
            #     continue # TODO fix this
            
            accepted_total += 1
            sum_top1_conf += top1_prob
            sum_probs += prob_vec

            crop_path = crops_dir / f"det_{accepted_total:05d}.jpg"
            to_save = transforms_to_pil(images_t[i].cpu())
            to_save.save(crop_path)

            grid_items.append(GridItem(path=crop_path, label=id_to_name(top1_id), confidence=top1_prob))
            update_grid()

            summary = log_summary()
            event = {
                "ts_unix": int(time.time()),
                "screen_path": str(screen_path) if cfg.output.save_screenshots else "",
                "det_conf": float(det["conf"]),
                "det_cls": float(det["cls"]),
                "bbox_px": [det["x1"], det["y1"], det["x2"], det["y2"]],
                "bbox_norm": bbox_norm,
                "pred": {
                    "country_id": top1_id,
                    "country": id_to_name(top1_id),
                    "conf": top1_prob,
                },
                "topk": topk_list,
                "avg_confidence": summary["avg_confidence"],
                "top3_avg_prob": summary["top3_avg_prob"],
                "crop_path": str(crop_path),
            }
            jsonl_f.write(json.dumps(event) + "\n")
            jsonl_f.flush()
            logger.info(
                "Accepted: %s (%.2f) avg=%.3f",
                id_to_name(top1_id),
                top1_prob,
                summary["avg_confidence"],
            )
            if summary["top3_avg_prob"]:
                top3_str = ", ".join(
                    f"{row['country']}:{row['avg_prob']:.2f}" for row in summary["top3_avg_prob"]
                )
                logger.info("Top3 avg prob: %s", top3_str)

    def transforms_to_pil(img_t: torch.Tensor) -> Image.Image:
        from bollards.data.transforms import denormalize

        img = denormalize(img_t).clamp(0.0, 1.0)
        arr = (img.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
        return Image.fromarray(arr)

    def capture_and_handle(sct) -> None:
        screen = _capture_screen(sct, cfg)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        screen_path = screenshots_dir / f"screen_{ts}.jpg"
        screen.save(screen_path)
        logger.info("Screenshot captured and saved to %s", screen_path)
        handle_capture(screen, screen_path)
        if not cfg.output.save_screenshots:
            screen_path.unlink(missing_ok=True)

    logger.info("Live screen session started. Output: %s", run_dir)

    try:
        from mss import mss
    except Exception:
        raise SystemExit("Missing mss. Install requirements.live.txt")

    listener = None
    try:
        if cfg.trigger.mode == "stdin":
            logger.info("Trigger mode: stdin")
            with mss() as sct:
                while True:
                    try:
                        prompt = cfg.trigger.stdin_prompt + " "
                        cmd = input(prompt)
                    except (EOFError, KeyboardInterrupt):
                        break
                    if cmd.strip().lower() in {"q", "quit", "exit"}:
                        break
                    capture_and_handle(sct)
        else:
            logger.info("Trigger mode: hotkey (%s)", cfg.trigger.hotkey)
            trigger_queue: queue.Queue = queue.Queue()
            listener = _setup_hotkey_listener(cfg.trigger.hotkey, trigger_queue, logger)
            if listener is None:
                logger.info("Falling back to stdin mode")
                with mss() as sct:
                    while True:
                        try:
                            prompt = cfg.trigger.stdin_prompt + " "
                            cmd = input(prompt)
                        except (EOFError, KeyboardInterrupt):
                            break
                        if cmd.strip().lower() in {"q", "quit", "exit"}:
                            break
                        capture_and_handle(sct)
            else:
                with mss() as sct:
                    while True:
                        try:
                            _ = trigger_queue.get(timeout=0.25)
                        except queue.Empty:
                            continue
                        except KeyboardInterrupt:
                            break
                        capture_and_handle(sct)
    finally:
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                pass

        summary_path = run_dir / "summary.json"
        summary = log_summary()
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        jsonl_f.close()
        if viewer is not None:
            viewer.close()

        logger.info("Session ended. Saved summary to %s", summary_path)
