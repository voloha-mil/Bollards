from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from utils import ensure_dir


def s3_key(prefix: str, *parts: str) -> str:
    p = prefix.strip("/")
    tail = "/".join(x.strip("/").replace("\\", "/") for x in parts if x)
    return f"{p}/{tail}" if p else tail


@lru_cache(maxsize=1)
def _s3_client():
    import boto3
    return boto3.client("s3")


def s3_upload_file(*, bucket: str, key: str, local_path: Path, logger: Optional[logging.Logger] = None) -> None:
    if not local_path.exists():
        return
    _s3_client().upload_file(str(local_path), bucket, key)
    if logger:
        logger.info("S3 upload: s3://%s/%s", bucket, key)


def s3_download_file_if_exists(
    *, bucket: str, key: str, local_path: Path, logger: Optional[logging.Logger] = None
) -> bool:
    ensure_dir(local_path.parent)
    try:
        _s3_client().download_file(bucket, key, str(local_path))
        if logger:
            logger.info("S3 download: s3://%s/%s -> %s", bucket, key, local_path)
        return True
    except Exception:
        return False


def restore_state_from_s3(*, bucket: str, prefix: str, split: str, filtered_dir: Path, logger: logging.Logger) -> None:
    s3_download_file_if_exists(
        bucket=bucket,
        key=s3_key(prefix, split, "state/cursor.json"),
        local_path=filtered_dir / "cursor.json",
        logger=logger,
    )
    s3_download_file_if_exists(
        bucket=bucket,
        key=s3_key(prefix, split, "state/processed_ids.txt"),
        local_path=filtered_dir / "processed_ids.txt",
        logger=logger,
    )
    s3_download_file_if_exists(
        bucket=bucket,
        key=s3_key(prefix, split, "state/filtered.csv"),
        local_path=filtered_dir / "filtered.csv",
        logger=logger,
    )


def sync_state_to_s3(*, bucket: str, prefix: str, split: str, filtered_dir: Path, logger: logging.Logger) -> None:
    s3_upload_file(
        bucket=bucket,
        key=s3_key(prefix, split, "state/cursor.json"),
        local_path=filtered_dir / "cursor.json",
        logger=logger,
    )
    s3_upload_file(
        bucket=bucket,
        key=s3_key(prefix, split, "state/processed_ids.txt"),
        local_path=filtered_dir / "processed_ids.txt",
        logger=logger,
    )
    s3_upload_file(
        bucket=bucket,
        key=s3_key(prefix, split, "state/filtered.csv"),
        local_path=filtered_dir / "filtered.csv",
        logger=logger,
    )
