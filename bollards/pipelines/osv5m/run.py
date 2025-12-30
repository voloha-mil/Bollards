from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from bollards.pipelines.osv5m.config import MinerConfig
from bollards.constants import OSV5M_FILTERED_FIELDS_PREFIX, OSV5M_FILTERED_FIELDS_SUFFIX
from bollards.models.detector_yolo import (
    extract_detection_payload,
    load_yolo,
    run_inference_batch,
    save_annotated,
)
from bollards.utils.io.csv import append_csv_row, append_processed_ids, ensure_csv_header, load_processed_ids
from bollards.utils.io.fs import ensure_dir
from bollards.utils.io.hf import hf_download_model_file
from bollards.pipelines.osv5m.common import (
    HF_MODEL_FILENAME,
    HF_MODEL_REPO,
    META_FIELDS,
    ShardCache,
    shard_ids_for_split,
)
from bollards.pipelines.osv5m.data import (
    close_shard,
    load_cursor,
    load_metadata_map,
    load_next_batch,
    save_cursor,
)
from bollards.pipelines.osv5m.s3 import (
    restore_state_from_s3,
    s3_key,
    s3_upload_file,
    sync_state_to_s3,
)


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("osv5m_miner")


def meta_to_row(meta) -> dict[str, object]:
    return {k: (getattr(meta, k) if meta else "") for k in META_FIELDS}


def process_batch(
    *,
    split: str,
    imgsz: int,
    conf: float,
    device: str,
    batch_size: int,
    logger: logging.Logger,
    model,
    batch,
    out_images: Path,
    out_annot: Path,
    filtered_csv: Path,
    fieldnames: list[str],
    meta_map,
    positives_remaining: int,
    processed: set[str],
    processed_txt: Path,
    s3_bucket: Optional[str],
    s3_prefix: str,
    s3_delete_local: bool,
) -> tuple[int, bool]:
    if positives_remaining <= 0:
        return 0, True

    try:
        results_list = run_inference_batch(
            model=model,
            image_paths=batch.extracted_paths,
            imgsz=imgsz,
            conf=conf,
            device=device,
            batch=batch_size,
        )
    except Exception as e:
        logger.warning("Inference failed for batch shard=%s: %s", batch.shard, str(e))
        for p in batch.extracted_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        return 0, False

    ids_to_mark: list[str] = []
    added = 0

    for img_path, r0 in tqdm(list(zip(batch.extracted_paths, results_list)), desc=f"post {split}/{batch.shard}"):
        img_id = img_path.stem
        ids_to_mark.append(img_id)

        payload = extract_detection_payload(r0)
        if payload.n_boxes <= 0:
            img_path.unlink(missing_ok=True)
            continue

        if added >= positives_remaining:
            img_path.unlink(missing_ok=True)
            continue

        final_img_path = out_images / f"{img_id}.jpg"
        shutil.move(str(img_path), str(final_img_path))

        annot_path = out_annot / f"{img_id}.jpg"
        save_annotated(r0, annot_path)

        meta = meta_map.get(img_id) if meta_map else None

        s3_image_key = ""
        s3_annot_key = ""
        image_path_for_csv = str(final_img_path)
        annot_path_for_csv = str(annot_path)

        if s3_bucket:
            s3_image_key = s3_key(s3_prefix, split, "images", f"{img_id}.jpg")
            s3_annot_key = s3_key(s3_prefix, split, "annotated", f"{img_id}.jpg")
            s3_upload_file(bucket=s3_bucket, key=s3_image_key, local_path=final_img_path, logger=None)
            s3_upload_file(bucket=s3_bucket, key=s3_annot_key, local_path=annot_path, logger=None)

            if s3_delete_local:
                final_img_path.unlink(missing_ok=True)
                annot_path.unlink(missing_ok=True)
                image_path_for_csv = ""
                annot_path_for_csv = ""

        row = {
            "id": img_id,
            "split": split,
            "shard": batch.shard,
            "image_path": image_path_for_csv,
            "annotated_path": annot_path_for_csv,
            **meta_to_row(meta),
            "n_boxes": payload.n_boxes,
            "boxes_xyxy": payload.boxes_xyxy_json,
            "boxes_conf": payload.boxes_conf_json,
            "boxes_cls": payload.boxes_cls_json,
            "model_repo": HF_MODEL_REPO,
            "model_file": HF_MODEL_FILENAME,
            "created_at_unix": int(time.time()),
            "s3_bucket": s3_bucket or "",
            "s3_prefix": s3_prefix or "",
            "s3_image_key": s3_image_key,
            "s3_annotated_key": s3_annot_key,
        }
        append_csv_row(filtered_csv, fieldnames, row)
        added += 1

    new_ids = [i for i in ids_to_mark if i not in processed]
    if new_ids:
        processed.update(new_ids)
        append_processed_ids(processed_txt, new_ids)

    return added, True


def run_miner(cfg: MinerConfig) -> int:
    logger = setup_logger()

    if cfg.worker_id < 0 or cfg.worker_id >= cfg.num_workers:
        raise SystemExit("worker_id must be in [0, num_workers)")

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    hf_cache = Path(cfg.hf_cache)
    workdir = Path(cfg.workdir)
    filtered = Path(cfg.filtered)
    extracted_dir = workdir / "extracted"

    out_images = filtered / "images"
    out_annot = filtered / "annotated"
    ensure_dir(out_images)
    ensure_dir(out_annot)

    filtered_csv = filtered / "filtered.csv"
    processed_txt = filtered / "processed_ids.txt"
    cursor_path = filtered / "cursor.json"

    fieldnames = OSV5M_FILTERED_FIELDS_PREFIX + META_FIELDS + OSV5M_FILTERED_FIELDS_SUFFIX
    ensure_csv_header(filtered_csv, fieldnames)

    if cfg.s3_bucket and cfg.s3_restore:
        restore_state_from_s3(
            bucket=cfg.s3_bucket,
            prefix=cfg.s3_prefix,
            split=cfg.split,
            filtered_dir=filtered,
            logger=logger,
        )

    processed = load_processed_ids(processed_txt)
    cursor = load_cursor(cursor_path)
    shard_cache = ShardCache()

    meta_map = load_metadata_map(cache_dir=hf_cache, meta_mode=cfg.meta, logger=logger)

    weights_path = hf_download_model_file(repo_id=HF_MODEL_REPO, filename=HF_MODEL_FILENAME, cache_dir=hf_cache)
    logger.info("Using model weights: %s", weights_path)
    model = load_yolo(weights_path)

    shard_ids_all = shard_ids_for_split(cfg.split)
    shard_ids = [s for i, s in enumerate(shard_ids_all) if i % cfg.num_workers == cfg.worker_id]
    logger.info(
        "worker=%d/%d scanning %d shard(s) in split=%s",
        cfg.worker_id,
        cfg.num_workers,
        len(shard_ids),
        cfg.split,
    )

    positives = 0
    batches = 0
    t0 = time.time()

    last_fail_key: Optional[tuple[int, int]] = None
    fail_count = 0

    try:
        while positives < cfg.target:
            batch = load_next_batch(
                split=cfg.split,
                batch_size=cfg.batch,
                logger=logger,
                shard_ids=shard_ids,
                hf_cache=hf_cache,
                extracted_dir=extracted_dir,
                processed=processed,
                cursor=cursor,
                cursor_path=cursor_path,
                shard_cache=shard_cache,
            )

            if batch is None:
                if cursor.shard_index >= len(shard_ids):
                    logger.info("Reached end of split=%s. No more shards.", cfg.split)
                    break
                continue

            fail_key = (batch.cursor_before.shard_index, batch.cursor_before.member_index)

            added, ok = process_batch(
                split=cfg.split,
                imgsz=cfg.imgsz,
                conf=cfg.conf,
                device=cfg.device,
                batch_size=cfg.batch,
                logger=logger,
                model=model,
                batch=batch,
                out_images=out_images,
                out_annot=out_annot,
                filtered_csv=filtered_csv,
                fieldnames=fieldnames,
                meta_map=meta_map,
                positives_remaining=(cfg.target - positives),
                processed=processed,
                processed_txt=processed_txt,
                s3_bucket=cfg.s3_bucket,
                s3_prefix=cfg.s3_prefix,
                s3_delete_local=cfg.s3_delete_local,
            )

            if not ok:
                if last_fail_key == fail_key:
                    fail_count += 1
                else:
                    last_fail_key = fail_key
                    fail_count = 1

                if fail_count <= cfg.max_retries:
                    cursor.shard_index = batch.cursor_before.shard_index
                    cursor.member_index = batch.cursor_before.member_index
                    save_cursor(cursor_path, cursor)
                    continue

                last_fail_key = None
                fail_count = 0

            positives += added
            batches += 1

            if cfg.s3_bucket and cfg.s3_sync_state_every > 0 and (batches % cfg.s3_sync_state_every == 0):
                sync_state_to_s3(
                    bucket=cfg.s3_bucket,
                    prefix=cfg.s3_prefix,
                    split=cfg.split,
                    filtered_dir=filtered,
                    logger=logger,
                )

            elapsed = time.time() - t0
            logger.info(
                "positives: %d/%d | elapsed: %.1fs | cursor: shard=%d member=%d",
                positives,
                cfg.target,
                elapsed,
                cursor.shard_index,
                cursor.member_index,
            )

    finally:
        close_shard(shard_cache)
        if extracted_dir.exists():
            shutil.rmtree(extracted_dir, ignore_errors=True)

    if cfg.s3_bucket:
        sync_state_to_s3(
            bucket=cfg.s3_bucket,
            prefix=cfg.s3_prefix,
            split=cfg.split,
            filtered_dir=filtered,
            logger=logger,
        )

    logger.info("Done. Filtered dir: %s", filtered.resolve())
    return 0
