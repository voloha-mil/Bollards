from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import numpy as np
import torch


def _derive_seed(base_seed: int, name: str) -> int:
    payload = f"{int(base_seed)}:{name}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "little")


@dataclass(frozen=True)
class SeedBundle:
    seed: int
    python_seed: int
    numpy_seed: int
    torch_seed: int


def seed_everything(seed: int, deterministic: bool = True) -> SeedBundle:
    python_seed = _derive_seed(seed, "python")
    numpy_seed = _derive_seed(seed, "numpy")
    torch_seed = _derive_seed(seed, "torch")

    random.seed(python_seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass

    return SeedBundle(
        seed=int(seed),
        python_seed=python_seed,
        numpy_seed=numpy_seed,
        torch_seed=torch_seed,
    )


def make_python_rng(seed: int, name: str) -> random.Random:
    return random.Random(_derive_seed(seed, name))


def make_numpy_rng(seed: int, name: str) -> np.random.Generator:
    return np.random.default_rng(_derive_seed(seed, name))


def make_torch_generator(seed: int, name: str) -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(_derive_seed(seed, name))
    return gen


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
