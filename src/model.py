from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ARCHS = {
    "unet": "Unet",
    "unetplusplus": "UnetPlusPlus",
    "deeplabv3plus": "DeepLabV3Plus",
    "fpn": "FPN",
    "pspnet": "PSPNet",
}


def build_model(arch: str = "unet", encoder: str = "resnet34", num_classes: int = 4,
                encoder_weights: str | None = "imagenet") -> nn.Module:
    import segmentation_models_pytorch as smp

    key = arch.lower().replace("-", "").replace("_", "")
    if key not in ARCHS:
        raise ValueError(f"Неизвестная архитектура {arch}; доступны: {list(ARCHS)}")
    cls = getattr(smp, ARCHS[key])
    return cls(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=3, classes=num_classes)


class DiceCELoss(nn.Module):

    def __init__(self, num_classes: int, class_weights: torch.Tensor | None = None,
                 dice_weight: float = 1.0, ce_weight: float = 1.0, smooth: float = 1.0):
        super().__init__()
        self.n = num_classes
        self.register_buffer("w", class_weights if class_weights is not None else torch.ones(num_classes))
        self.dice_weight, self.ce_weight, self.smooth = dice_weight, ce_weight, smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.w.to(logits.dtype))
        probs = logits.float().softmax(1)
        onehot = F.one_hot(target, self.n).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)
        card = probs.sum(dims) + onehot.sum(dims)
        dice = (2 * inter + self.smooth) / (card + self.smooth)
        present = onehot.sum(dims) > 0
        present[0] = True
        dice_loss = 1 - dice[present].mean()
        return self.ce_weight * ce + self.dice_weight * dice_loss


def compute_class_weights(pixel_counts: list[int] | torch.Tensor, max_w: float = 10.0) -> torch.Tensor:
    c = torch.as_tensor(pixel_counts, dtype=torch.float64).clamp(min=1)
    freq = c / c.sum()
    w = 1.0 / freq.sqrt()
    w = w / w[0]
    return w.clamp(max=max_w).float()
