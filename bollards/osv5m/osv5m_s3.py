from __future__ import annotations

import logging
from pathlib import Path

from bollards.io.s3 import s3_download_file_if_exists, s3_key, s3_upload_file


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
