from __future__ import annotations

import argparse
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from detector import extract_detection_payload, load_yolo, run_inference_batch, save_annotated
from osv5m import ShardCache, shard_ids_for_split, META_FIELDS, HF_MODEL_REPO, HF_MODEL_FILENAME
from osv5m import load_cursor, load_metadata_map, load_next_batch, save_cursor, close_shard
from osv5m import restore_state_from_s3, sync_state_to_s3, s3_key, s3_upload_file
from utils import (
    append_csv_row,
    append_processed_ids,
    ensure_csv_header,
    ensure_dir,
    hf_download_model_file,
    load_processed_ids,
)


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("osv5m_miner")


def meta_to_row(meta) -> dict[str, object]:
    # meta may be None
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

    # IMPORTANT: only mark processed AFTER inference succeeded
    new_ids = [i for i in ids_to_mark if i not in processed]
    if new_ids:
        processed.update(new_ids)
        append_processed_ids(processed_txt, new_ids)

    return added, True


def main() -> int:
    logger = setup_logger()

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "train"], default="test")
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="auto")  # on EC2 GPU use: 0
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)

    ap.add_argument("--hf-cache", default="./hf_cache")
    ap.add_argument("--workdir", default="./work_tmp")
    ap.add_argument("--filtered", default="./filtered_dataset_osv5m")
    ap.add_argument("--meta", choices=["none", "test", "train"], default="test")

    ap.add_argument("--s3-bucket", default=None)
    ap.add_argument("--s3-prefix", default="")
    ap.add_argument("--s3-restore", action="store_true")
    ap.add_argument("--s3-sync-state-every", type=int, default=1)
    ap.add_argument("--s3-delete-local", action="store_true")

    ap.add_argument("--max-retries", type=int, default=2)

    args = ap.parse_args()

    # mac safety; harmless on EC2 CUDA
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    hf_cache = Path(args.hf_cache)
    workdir = Path(args.workdir)
    filtered = Path(args.filtered)
    extracted_dir = workdir / "extracted"

    out_images = filtered / "images"
    out_annot = filtered / "annotated"
    ensure_dir(out_images)
    ensure_dir(out_annot)

    filtered_csv = filtered / "filtered.csv"
    processed_txt = filtered / "processed_ids.txt"
    cursor_path = filtered / "cursor.json"

    fieldnames = [
        "id",
        "split",
        "shard",
        "image_path",
        "annotated_path",
        *META_FIELDS,
        "n_boxes",
        "boxes_xyxy",
        "boxes_conf",
        "boxes_cls",
        "model_repo",
        "model_file",
        "created_at_unix",
        "s3_bucket",
        "s3_prefix",
        "s3_image_key",
        "s3_annotated_key",
    ]
    ensure_csv_header(filtered_csv, fieldnames)

    if args.s3_bucket and args.s3_restore:
        restore_state_from_s3(
            bucket=args.s3_bucket,
            prefix=args.s3_prefix,
            split=args.split,
            filtered_dir=filtered,
            logger=logger,
        )

    processed = load_processed_ids(processed_txt)
    cursor = load_cursor(cursor_path)
    shard_cache = ShardCache()

    meta_map = load_metadata_map(cache_dir=hf_cache, meta_mode=args.meta, logger=logger)

    weights_path = hf_download_model_file(repo_id=HF_MODEL_REPO, filename=HF_MODEL_FILENAME, cache_dir=hf_cache)
    logger.info("Using model weights: %s", weights_path)
    model = load_yolo(weights_path)

    shard_ids = shard_ids_for_split(args.split)

    positives = 0
    batches = 0
    t0 = time.time()

    last_fail_key: Optional[tuple[int, int]] = None
    fail_count = 0

    try:
        while positives < args.target:
            batch = load_next_batch(
                split=args.split,
                batch_size=args.batch,
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
                    logger.info("Reached end of split=%s. No more shards.", args.split)
                    break
                continue

            fail_key = (batch.cursor_before.shard_index, batch.cursor_before.member_index)

            added, ok = process_batch(
                split=args.split,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                batch_size=args.batch,
                logger=logger,
                model=model,
                batch=batch,
                out_images=out_images,
                out_annot=out_annot,
                filtered_csv=filtered_csv,
                fieldnames=fieldnames,
                meta_map=meta_map,
                positives_remaining=(args.target - positives),
                processed=processed,
                processed_txt=processed_txt,
                s3_bucket=args.s3_bucket,
                s3_prefix=args.s3_prefix,
                s3_delete_local=args.s3_delete_local,
            )

            if not ok:
                if last_fail_key == fail_key:
                    fail_count += 1
                else:
                    last_fail_key = fail_key
                    fail_count = 1

                if fail_count <= args.max_retries:
                    cursor.shard_index = batch.cursor_before.shard_index
                    cursor.member_index = batch.cursor_before.member_index
                    save_cursor(cursor_path, cursor)
                    continue

                last_fail_key = None
                fail_count = 0

            positives += added
            batches += 1

            if args.s3_bucket and args.s3_sync_state_every > 0 and (batches % args.s3_sync_state_every == 0):
                sync_state_to_s3(bucket=args.s3_bucket, prefix=args.s3_prefix, split=args.split, filtered_dir=filtered, logger=logger)

            elapsed = time.time() - t0
            logger.info(
                "positives: %d/%d | elapsed: %.1fs | cursor: shard=%d member=%d",
                positives,
                args.target,
                elapsed,
                cursor.shard_index,
                cursor.member_index,
            )

    finally:
        close_shard(shard_cache)
        if extracted_dir.exists():
            shutil.rmtree(extracted_dir, ignore_errors=True)

    if args.s3_bucket:
        sync_state_to_s3(bucket=args.s3_bucket, prefix=args.s3_prefix, split=args.split, filtered_dir=filtered, logger=logger)

    logger.info("Done. Filtered dir: %s", filtered.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
