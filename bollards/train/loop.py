import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def _average_precision(scores: torch.Tensor, targets: torch.Tensor) -> float:
    if scores.numel() == 0:
        return 0.0
    order = torch.argsort(scores, descending=True)
    sorted_targets = targets[order].float()
    total_pos = float(sorted_targets.sum().item())
    if total_pos == 0.0:
        return float("nan")
    ranks = torch.arange(1, sorted_targets.numel() + 1, device=sorted_targets.device).float()
    precision = sorted_targets.cumsum(0) / ranks
    ap = (precision * sorted_targets).sum() / total_pos
    return float(ap.item())


def _mean_average_precision(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if logits.numel() == 0:
        return 0.0
    logits = logits.float()
    labels = labels.long()
    num_classes = logits.size(1)
    ap_values = []
    for cls in range(num_classes):
        targets = labels == cls
        if not targets.any():
            continue
        ap = _average_precision(logits[:, cls], targets)
        if not math.isnan(ap):
            ap_values.append(ap)
    if not ap_values:
        return 0.0
    return float(sum(ap_values) / len(ap_values))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "val",
) -> Tuple[float, float, float]:
    model.eval()
    correct1 = 0
    correct5 = 0
    total = 0
    all_logits = []
    all_labels = []

    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    with torch.no_grad():
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

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

            pbar.set_postfix_str(
                f"top1={correct1/max(total,1):.3f} top5={correct5/max(total,1):.3f}"
            )

    if all_logits:
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        mean_ap = _mean_average_precision(logits, labels)
    else:
        mean_ap = 0.0

    return correct1 / max(total, 1), correct5 / max(total, 1), mean_ap


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
