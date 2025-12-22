import csv
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from huggingface_hub import hf_hub_download


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_processed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip())


def append_processed_ids(path: Path, ids: Iterable[str]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for i in ids:
            f.write(str(i) + "\n")


def ensure_csv_header(csv_path: Path, fieldnames: List[str]) -> None:
    if csv_path.exists():
        return
    ensure_dir(csv_path.parent)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()


def append_csv_row(csv_path: Path, fieldnames: List[str], row: Dict) -> None:
    ensure_dir(csv_path.parent)
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow({k: row.get(k, "") for k in fieldnames})


def hf_download_dataset_file(
    *,
    repo_id: str,
    filename: str,
    cache_dir: Path,
    subfolder: Optional[str] = None,
) -> Path:
    local = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        subfolder=subfolder,
        cache_dir=str(cache_dir),
    )
    return Path(local)


def hf_download_model_file(
    *,
    repo_id: str,
    filename: str,
    cache_dir: Path,
) -> Path:
    local = hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename=filename,
        cache_dir=str(cache_dir),
    )
    return Path(local)


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
    *,
    bucket: str,
    key: str,
    local_path: Path,
    logger: Optional[logging.Logger] = None,
) -> bool:
    ensure_dir(local_path.parent)
    try:
        _s3_client().download_file(bucket, key, str(local_path))
        if logger:
            logger.info("S3 download: s3://%s/%s -> %s", bucket, key, local_path)
        return True
    except Exception:
        return False
