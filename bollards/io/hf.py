from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download


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
