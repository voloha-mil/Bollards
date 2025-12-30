from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from bollards.pipelines.train.config import TrainConfig
from bollards.utils.io.hf import hf_upload_run_artifacts


def _sanitize_config(cfg: TrainConfig) -> dict:
    data = asdict(cfg)
    hub_cfg = data.get("hub")
    if isinstance(hub_cfg, dict) and hub_cfg.get("token"):
        hub_cfg["token"] = "REDACTED"
    return data


def _format_template(template: Optional[str], **kwargs: object) -> Optional[str]:
    if not template:
        return template
    if "{" not in template:
        return template
    try:
        return template.format(**kwargs)
    except KeyError as exc:
        raise ValueError(f"Unknown placeholder in template: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"Invalid template format: {exc}") from exc


def _maybe_push_best_to_hf(
    cfg: TrainConfig,
    *,
    run_dir: str,
    run_name: str,
    best_metric_name: str,
    best_metric_value: float,
) -> None:
    if not cfg.hub.enabled:
        return

    repo_id = (cfg.hub.repo_id or "").strip()
    if not repo_id:
        raise ValueError("hub.enabled requires hub.repo_id")

    include = list(cfg.hub.upload_include or [])
    for required in ("best.pt", "config.json"):
        if required not in include:
            include.append(required)

    run_dir_path = Path(run_dir)
    best_path = run_dir_path / "best.pt"
    if not best_path.exists():
        print("[warn] best.pt not found; skipping Hugging Face upload.")
        return

    token = cfg.hub.token
    if not token and cfg.hub.token_env:
        token = os.getenv(cfg.hub.token_env)
        if not token:
            print(
                f"[warn] Hugging Face token env '{cfg.hub.token_env}' not set; "
                "falling back to cached credentials."
            )

    template_args = {
        "run_name": run_name,
        "best_metric": best_metric_name,
        "best_metric_value": best_metric_value,
    }
    path_in_repo = _format_template(cfg.hub.path_in_repo, **template_args)
    commit_message = _format_template(cfg.hub.commit_message, **template_args)

    uploaded = hf_upload_run_artifacts(
        repo_id=repo_id,
        run_dir=run_dir_path,
        allow_patterns=include,
        path_in_repo=path_in_repo,
        private=cfg.hub.private,
        commit_message=commit_message,
        token=token,
    )
    if uploaded:
        print(f"[info] uploaded {len(uploaded)} artifact(s) to huggingface.co/{repo_id}")
    else:
        print("[warn] no artifacts matched for Hugging Face upload.")
