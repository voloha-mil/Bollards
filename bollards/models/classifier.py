from dataclasses import dataclass

import torch
import torch.nn as nn
import timm


@dataclass
class ModelConfig:
    backbone_name: str = "efficientnet_b1"
    pretrained: bool = True

    num_classes: int = 200  # set from config
    img_emb_dropout: float = 0.2

    # conditioning (bbox/conf) branch
    meta_dim: int = 5               # e.g., x_center, y_center, w, h, conf
    meta_hidden: int = 64
    meta_dropout: float = 0.0

    # head
    head_hidden: int = 256
    head_layers: int = 1
    head_dropout: float = 0.2


class BollardNet(nn.Module):
    """
    CNN backbone + custom head + tiny MLP conditioning on bbox geometry/confidence.

    Forward:
      logits = model(images, meta)
      images: (B, 3, H, W)
      meta:   (B, meta_dim) float32
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Create backbone with no classifier head; returns pooled feature vector.
        # timm: num_classes=0 makes it a feature extractor.
        self.backbone = timm.create_model(
            cfg.backbone_name,
            pretrained=cfg.pretrained,
            num_classes=0,
            global_pool="avg",
        )

        feat_dim = self._infer_feat_dim()

        self.meta_mlp = nn.Sequential(
            nn.Linear(cfg.meta_dim, cfg.meta_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cfg.meta_dropout),
        )

        self.head = self._build_head(feat_dim)

    def _build_head(self, feat_dim: int) -> nn.Sequential:
        if self.cfg.head_layers < 0:
            raise ValueError("head_layers must be >= 0")

        layers: list[nn.Module] = [nn.Dropout(p=self.cfg.img_emb_dropout)]
        in_dim = feat_dim + self.cfg.meta_hidden
        for _ in range(self.cfg.head_layers):
            layers.append(nn.Linear(in_dim, self.cfg.head_hidden))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=self.cfg.head_dropout))
            in_dim = self.cfg.head_hidden
        layers.append(nn.Linear(in_dim, self.cfg.num_classes))
        return nn.Sequential(*layers)

    def _infer_feat_dim(self) -> int:
        if hasattr(self.backbone, "num_features"):
            return int(self.backbone.num_features)  # type: ignore[attr-defined]
        # fallback: run a dummy forward
        with torch.no_grad():
            x = torch.zeros(1, 3, 224, 224)
            y = self.backbone(x)
            return int(y.shape[-1])

    def forward(self, images: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        if meta.dtype != torch.float32:
            meta = meta.float()

        img_feat = self.backbone(images)           # (B, feat_dim)
        meta_feat = self.meta_mlp(meta)            # (B, meta_hidden)
        fused = torch.cat([img_feat, meta_feat], dim=1)
        logits = self.head(fused)
        return logits

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True
