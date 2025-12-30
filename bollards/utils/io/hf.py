from pathlib import Path
from typing import Iterable, Optional

from huggingface_hub import HfApi, create_repo, hf_hub_download


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


def hf_upload_run_artifacts(
    *,
    repo_id: str,
    run_dir: Path,
    allow_patterns: Iterable[str],
    path_in_repo: Optional[str] = None,
    private: bool = False,
    commit_message: Optional[str] = None,
    token: Optional[str] = None,
) -> list[str]:
    patterns = [p for p in allow_patterns if p]
    if not patterns:
        return []

    matched: list[str] = []
    for pattern in patterns:
        for path in run_dir.glob(pattern):
            if path.is_file():
                matched.append(path.relative_to(run_dir).as_posix())
            elif path.is_dir():
                for sub_path in path.rglob("*"):
                    if sub_path.is_file():
                        matched.append(sub_path.relative_to(run_dir).as_posix())

    if not matched:
        return []

    matched = sorted(set(matched))

    create_repo(
        repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
        token=token,
    )

    api = HfApi()
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(run_dir),
        path_in_repo=path_in_repo,
        allow_patterns=matched,
        commit_message=commit_message,
        token=token,
    )
    return matched
