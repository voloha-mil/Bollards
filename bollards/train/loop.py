from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "val",
) -> Tuple[float, float]:
    model.eval()
    correct1 = 0
    correct5 = 0
    total = 0

    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        meta = batch["meta"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = model(images, meta)
        total += labels.size(0)

        pred1 = logits.argmax(dim=1)
        correct1 += (pred1 == labels).sum().item()

        top5 = torch.topk(logits, k=min(5, logits.size(1)), dim=1).indices
        correct5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

        pbar.set_postfix_str(f"top1={correct1/max(total,1):.3f} top5={correct5/max(total,1):.3f}")

    return correct1 / max(total, 1), correct5 / max(total, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    criterion: nn.Module,
    conf_weight_min: float,
) -> float:
    model.train()
    running = 0.0
    n = 0
    use_amp = scaler is not None

    pbar = tqdm(loader, desc="train", leave=False, dynamic_ncols=True)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        meta = batch["meta"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(images, meta)
                loss_vec = criterion(logits, labels)  # (B,)
        else:
            logits = model(images, meta)
            loss_vec = criterion(logits, labels)

        conf = meta[:, -1].detach()
        w = torch.clamp(conf, min=conf_weight_min, max=1.0)
        loss = (loss_vec * w).mean()

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running += float(loss.item()) * labels.size(0)
        n += labels.size(0)
        pbar.set_postfix_str(f"loss={running/max(n,1):.4f}")

    return running / max(n, 1)
