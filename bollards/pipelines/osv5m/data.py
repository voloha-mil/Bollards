from __future__ import annotations

import json
import logging
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from .common import Cursor, HF_DATASET_REPO, LoadedBatch, Meta, ShardCache
from bollards.utils.io.fs import ensure_dir
from bollards.utils.io.hf import hf_download_dataset_file


def load_metadata_map(cache_dir: Path, meta_mode: str, logger: logging.Logger) -> Optional[dict[str, Meta]]:
    """
    meta_mode:
      - "none": don't load metadata
      - "test": load test.csv
      - "train": load train.csv (large)
    """
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

    meta: dict[str, Meta] = {}
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


def close_shard(cache: ShardCache) -> None:
    try:
        if cache.zf is not None:
            cache.zf.close()
    except Exception:
        pass
    cache.shard = None
    cache.zf = None
    cache.members = None


def zip_members_jpg(z: zipfile.ZipFile) -> List[str]:
    # preserves zip order
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


def load_next_batch(
    *,
    split: str,
    batch_size: int,
    logger: logging.Logger,
    shard_ids: List[str],
    hf_cache: Path,
    extracted_dir: Path,
    processed: set[str],
    cursor: Cursor,
    cursor_path: Path,
    shard_cache: ShardCache,
) -> Optional[LoadedBatch]:
    """
    - Opens/downloads shard only when shard changes (no repeated HF calls per batch)
    - Picks next consecutive unprocessed batch from the shard
    - Extracts images to extracted_dir
    - Advances cursor and saves it
    """
    if cursor.shard_index >= len(shard_ids):
        return None

    shard = shard_ids[cursor.shard_index]

    # open shard only if changed
    if shard_cache.shard != shard:
        close_shard(shard_cache)

        zip_name = f"{shard}.zip"
        zip_subfolder = f"images/{split}"
        logger.info("Opening shard %s (%d/%d) at member_index=%d",
                    zip_name, cursor.shard_index + 1, len(shard_ids), cursor.member_index)

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

    # shard exhausted -> next shard
    if cursor.member_index >= len(shard_cache.members):
        close_shard(shard_cache)
        cursor.shard_index += 1
        cursor.member_index = 0
        save_cursor(cursor_path, cursor)
        return None

    cursor_before = Cursor(cursor.shard_index, cursor.member_index)

    picked, cursor = pick_next_batch_consecutive(
        members=shard_cache.members,
        processed=processed,
        cursor=cursor,
        batch_size=batch_size,
    )
    save_cursor(cursor_path, cursor)

    if not picked:
        return None

    if extracted_dir.exists():
        shutil.rmtree(extracted_dir, ignore_errors=True)
    ensure_dir(extracted_dir)

    extracted_paths = extract_members_from_open_zip(shard_cache.zf, picked, extracted_dir)
    return LoadedBatch(shard=shard, extracted_paths=extracted_paths, cursor_before=cursor_before)
