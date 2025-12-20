import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from huggingface_hub import hf_hub_download


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_processed_ids(path: Path) -> Set[str]:
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
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow({k: row.get(k, "") for k in fieldnames})


def hf_download_dataset_file(
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
