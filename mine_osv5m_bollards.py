import argparse
import json
import logging
import os
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from detector import extract_detection_payload, load_yolo, run_inference_batch, save_annotated
from utils import (
    append_csv_row,
    append_processed_ids,
    ensure_csv_header,
    ensure_dir,
    hf_download_dataset_file,
    hf_download_model_file,
    load_processed_ids,
)

HF_DATASET_REPO = "osv5m/osv5m"
HF_MODEL_REPO = "maco018/YOLOv12_traffic-delineator"
HF_MODEL_FILENAME = "models/YOLOv12_traffic-delineator.pt"


@dataclass
class Meta:
    id: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    country: Optional[str] = None
    region: Optional[str] = None
    sub_region: Optional[str] = None
    city: Optional[str] = None
    thumb_original_url: Optional[str] = None
    captured_at: Optional[str] = None
    creator_username: Optional[str] = None
    creator_id: Optional[str] = None


META_FIELDS = [
    "latitude",
    "longitude",
    "country",
    "region",
    "sub_region",
    "city",
    "captured_at",
    "creator_username",
    "creator_id",
    "thumb_original_url",
]


@dataclass
class Cursor:
    shard_index: int = 0
    member_index: int = 0


@dataclass
class ShardCache:
    shard: Optional[str] = None
    zf: Optional[zipfile.ZipFile] = None
    members: Optional[List[str]] = None


@dataclass
class LoadedBatch:
    shard: str
    extracted_paths: List[Path]


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("osv5m_miner")


def shard_ids_for_split(split: str) -> List[str]:
    if split == "test":
        return [f"{i:02d}" for i in range(5)]
    if split == "train":
        return [f"{i:02d}" for i in range(98)]
    raise ValueError("split must be 'test' or 'train'")


def zip_members_jpg(z: zipfile.ZipFile) -> List[str]:
    return [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg")) and not n.endswith("/")]


def extract_members_from_open_zip(z: zipfile.ZipFile, members: List[str], out_dir: Path) -> List[Path]:
    ensure_dir(out_dir)
    extracted: List[Path] = []
    for m in members:
        target = out_dir / Path(m).name
        with z.open(m) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
        extracted.append(target)
    return extracted


def load_metadata_map(cache_dir: Path, meta_mode: str, logger: logging.Logger) -> Optional[Dict[str, Meta]]:
    if meta_mode == "none":
        return None
    if meta_mode not in ("test", "train"):
        raise ValueError("meta_mode must be one of: none, test, train")

    fname = "test.csv" if meta_mode == "test" else "train.csv"
    meta_path = hf_download_dataset_file(repo_id=HF_DATASET_REPO, filename=fname, cache_dir=cache_dir)
    logger.info("Loading metadata into memory: %s", meta_path)

    cols = pd.read_csv(meta_path, nrows=0).columns.tolist()
    wanted = [
        "id",
        "latitude",
        "longitude",
        "country",
        "region",
        "sub-region",
        "city",
        "thumb_original_url",
        "captured_at",
        "creator_username",
        "creator_id",
    ]
    usecols = [c for c in wanted if c in cols]
    df = pd.read_csv(meta_path, usecols=usecols, dtype={"id": str})

    if "sub-region" in df.columns:
        df = df.rename(columns={"sub-region": "sub_region"})
    df = df.set_index("id", drop=False)

    meta: Dict[str, Meta] = {}
    for _, r in df.iterrows():
        meta[str(r["id"])] = Meta(
            id=str(r.get("id")),
            latitude=float(r["latitude"]) if "latitude" in r and pd.notna(r["latitude"]) else None,
            longitude=float(r["longitude"]) if "longitude" in r and pd.notna(r["longitude"]) else None,
            country=str(r["country"]) if "country" in r and pd.notna(r["country"]) else None,
            region=str(r["region"]) if "region" in r and pd.notna(r["region"]) else None,
            sub_region=str(r["sub_region"]) if "sub_region" in r and pd.notna(r["sub_region"]) else None,
            city=str(r["city"]) if "city" in r and pd.notna(r["city"]) else None,
            thumb_original_url=str(r["thumb_original_url"]) if "thumb_original_url" in r and pd.notna(r["thumb_original_url"]) else None,
            captured_at=str(r["captured_at"]) if "captured_at" in r and pd.notna(r["captured_at"]) else None,
            creator_username=str(r["creator_username"]) if "creator_username" in r and pd.notna(r["creator_username"]) else None,
            creator_id=str(r["creator_id"]) if "creator_id" in r and pd.notna(r["creator_id"]) else None,
        )

    logger.info("Metadata loaded: %d rows", len(meta))
    return meta


def meta_to_row(meta: Optional[Meta]) -> Dict[str, object]:
    return {k: (getattr(meta, k) if meta else "") for k in META_FIELDS}


def load_cursor(path: Path) -> Cursor:
    if not path.exists():
        return Cursor()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Cursor(
            shard_index=int(data.get("shard_index", 0)),
            member_index=int(data.get("member_index", 0)),
        )
    except Exception:
        return Cursor()


def save_cursor(path: Path, cursor: Cursor) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps({"shard_index": cursor.shard_index, "member_index": cursor.member_index}, indent=2),
        encoding="utf-8",
    )


def pick_next_batch_consecutive(
    members: List[str],
    processed: set[str],
    cursor: Cursor,
    batch_size: int,
) -> Tuple[List[str], Cursor]:
    picked: List[str] = []
    i = cursor.member_index

    while i < len(members) and len(picked) < batch_size:
        m = members[i]
        img_id = Path(m).stem
        i += 1
        if img_id in processed:
            continue
        picked.append(m)

    cursor.member_index = i
    return picked, cursor


def close_shard(cache: ShardCache) -> None:
    if cache.zf is not None:
        cache.zf.close()
    cache.shard = None
    cache.zf = None
    cache.members = None


def load_next_batch(
    *,
    args: argparse.Namespace,
    logger: logging.Logger,
    shard_ids: List[str],
    hf_cache: Path,
    extracted_dir: Path,
    processed: set[str],
    processed_txt: Path,
    cursor: Cursor,
    cursor_path: Path,
    shard_cache: ShardCache,
) -> Optional[LoadedBatch]:
    """
    Loading part:
    - Opens/downloads a shard only when shard changes (no repeated HF calls per batch)
    - Picks next consecutive batch from current shard
    - Extracts images (batch) to extracted_dir
    - Marks processed early + persists cursor
    Returns: LoadedBatch or None (meaning "advance/shard exhausted/end")
    """

    if cursor.shard_index >= len(shard_ids):
        return None

    shard = shard_ids[cursor.shard_index]

    # Open shard only if shard changed
    if shard_cache.shard != shard:
        close_shard(shard_cache)

        zip_name = f"{shard}.zip"
        zip_subfolder = f"images/{args.split}"
        logger.info(
            "Opening shard %s (%d/%d) at member_index=%d",
            zip_name,
            cursor.shard_index + 1,
            len(shard_ids),
            cursor.member_index,
        )

        zip_path = hf_download_dataset_file(
            repo_id=HF_DATASET_REPO,
            filename=zip_name,
            cache_dir=hf_cache,
            subfolder=zip_subfolder,
        )

        zf = zipfile.ZipFile(zip_path, "r")
        members = zip_members_jpg(zf)

        if not members:
            logger.warning("No JPG members in %s. Skipping shard.", zip_path)
            zf.close()
            cursor.shard_index += 1
            cursor.member_index = 0
            save_cursor(cursor_path, cursor)
            return None

        shard_cache.shard = shard
        shard_cache.zf = zf
        shard_cache.members = members

    assert shard_cache.zf is not None and shard_cache.members is not None

    # If shard exhausted: advance shard and reset cache
    if cursor.member_index >= len(shard_cache.members):
        close_shard(shard_cache)
        cursor.shard_index += 1
        cursor.member_index = 0
        save_cursor(cursor_path, cursor)
        return None

    picked, cursor = pick_next_batch_consecutive(
        members=shard_cache.members,
        processed=processed,
        cursor=cursor,
        batch_size=args.batch,
    )
    save_cursor(cursor_path, cursor)

    if not picked:
        # likely skipped processed until end; next loop will either pick more or advance shard
        return None

    # Clean temp dir each loop to avoid growth
    if extracted_dir.exists():
        shutil.rmtree(extracted_dir, ignore_errors=True)
    ensure_dir(extracted_dir)

    extracted_paths = extract_members_from_open_zip(shard_cache.zf, picked, extracted_dir)

    # Mark processed early for resume safety
    for img_path in extracted_paths:
        img_id = img_path.stem
        if img_id not in processed:
            processed.add(img_id)
            append_processed_ids(processed_txt, [img_id])

    return LoadedBatch(shard=shard, extracted_paths=extracted_paths)


def process_batch(
    *,
    args: argparse.Namespace,
    logger: logging.Logger,
    model,
    batch: LoadedBatch,
    out_images: Path,
    out_annot: Path,
    filtered_csv: Path,
    fieldnames: List[str],
    meta_map: Optional[Dict[str, Meta]],
    positives_remaining: int,
) -> int:
    """
    Processing part:
    - Batched inference on extracted_paths
    - Keeps positives: move original, save annotated, append CSV
    - Deletes negatives from temp
    Returns number of new positives added.
    """
    if positives_remaining <= 0:
        return 0

    try:
        results_list = run_inference_batch(
            model=model,
            image_paths=batch.extracted_paths,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            batch=args.batch,
        )
    except Exception as e:
        logger.warning("Inference failed for batch shard=%s: %s", batch.shard, str(e))
        # best-effort cleanup temp files
        for p in batch.extracted_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        return 0

    added = 0

    for img_path, r0 in tqdm(
        list(zip(batch.extracted_paths, results_list)),
        desc=f"post {args.split}/{batch.shard}",
    ):
        if added >= positives_remaining:
            break

        img_id = img_path.stem

        payload = extract_detection_payload(r0)
        if payload.n_boxes <= 0:
            img_path.unlink(missing_ok=True)
            continue

        final_img_path = out_images / f"{img_id}.jpg"
        shutil.move(str(img_path), str(final_img_path))

        annot_path = out_annot / f"{img_id}.jpg"
        save_annotated(r0, annot_path)

        meta = meta_map.get(img_id) if meta_map else None

        row = {
            "id": img_id,
            "split": args.split,
            "shard": batch.shard,
            "image_path": str(final_img_path),
            "annotated_path": str(annot_path),
            **meta_to_row(meta),
            "n_boxes": payload.n_boxes,
            "boxes_xyxy": payload.boxes_xyxy_json,
            "boxes_conf": payload.boxes_conf_json,
            "boxes_cls": payload.boxes_cls_json,
            "model_repo": HF_MODEL_REPO,
            "model_file": HF_MODEL_FILENAME,
            "created_at_unix": int(time.time()),
        }
        append_csv_row(filtered_csv, fieldnames, row)

        added += 1

    return added


def main() -> int:
    logger = setup_logger()

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "train"], default="test", help="OSV split to mine")
    ap.add_argument("--target", type=int, required=True, help="Stop after collecting this many positives")
    ap.add_argument("--batch", type=int, default=16, help="Batch size (extract + inference per loop)")
    ap.add_argument("--device", default="auto", help="Ultralytics device: auto | cpu | mps | 0")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)

    ap.add_argument("--hf-cache", default="./hf_cache", help="HF cache directory for zips/csv/weights")
    ap.add_argument("--workdir", default="./work_tmp", help="Temp directory for extracted images")
    ap.add_argument("--filtered", default="./filtered_dataset_osv5m", help="Output directory for filtered dataset")

    ap.add_argument(
        "--meta",
        choices=["none", "test", "train"],
        default="test",
        help="Metadata loading mode. Recommended: test. Avoid: train unless you really want it.",
    )

    args = ap.parse_args()

    # Apple Silicon: allow MPS fallback for missing ops like torchvision::nms
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
    ]
    ensure_csv_header(filtered_csv, fieldnames)

    processed = load_processed_ids(processed_txt)
    cursor = load_cursor(cursor_path)
    shard_cache = ShardCache()

    meta_map = load_metadata_map(cache_dir=hf_cache, meta_mode=args.meta, logger=logger)

    weights_path = hf_download_model_file(
        repo_id=HF_MODEL_REPO,
        filename=HF_MODEL_FILENAME,
        cache_dir=hf_cache,
    )
    logger.info("Using model weights: %s", weights_path)
    model = load_yolo(weights_path)

    shard_ids = shard_ids_for_split(args.split)

    positives = 0
    t0 = time.time()

    try:
        while positives < args.target:
            batch = load_next_batch(
                args=args,
                logger=logger,
                shard_ids=shard_ids,
                hf_cache=hf_cache,
                extracted_dir=extracted_dir,
                processed=processed,
                processed_txt=processed_txt,
                cursor=cursor,
                cursor_path=cursor_path,
                shard_cache=shard_cache,
            )

            # No batch: could mean "end of shard" or "skipped processed" or "end of split"
            if batch is None:
                if cursor.shard_index >= len(shard_ids):
                    logger.info("Reached end of split=%s. No more shards to scan.", args.split)
                    break
                continue

            added = process_batch(
                args=args,
                logger=logger,
                model=model,
                batch=batch,
                out_images=out_images,
                out_annot=out_annot,
                filtered_csv=filtered_csv,
                fieldnames=fieldnames,
                meta_map=meta_map,
                positives_remaining=(args.target - positives),
            )
            positives += added

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

    logger.info("Filtered dataset written to: %s", filtered.resolve())
    logger.info("Table: %s", filtered_csv.resolve())
    logger.info("Cursor saved to: %s", cursor_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
