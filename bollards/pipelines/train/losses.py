from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float | list[float]] = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        if label_smoothing < 0 or label_smoothing > 1:
            raise ValueError("label_smoothing must be in [0, 1]")
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif alpha is None:
            self.alpha = None
        else:
            self.alpha = float(alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        logpt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()
        nll_loss = -logpt

        if self.label_smoothing > 0:
            smooth_loss = -log_probs.mean(dim=1)
            nll_loss = (1.0 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss

        if isinstance(self.alpha, torch.Tensor):
            alpha_t = self.alpha.to(logits.device, dtype=logits.dtype).gather(0, targets)
            nll_loss = nll_loss * alpha_t
        elif self.alpha is not None:
            nll_loss = nll_loss * float(self.alpha)

        loss = (1.0 - pt) ** self.gamma * nll_loss
        return loss
