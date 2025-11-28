# style_transfer/losses.py

import torch
from torch import nn
import torch.nn.functional as F


def gram_matrix(y: torch.Tensor) -> torch.Tensor:
    """
    y: [B, C, H, W] -> Gram: [B, C, C]
    """
    b, c, h, w = y.size()
    features = y.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))  # [B, C, C]
    return gram / (c * h * w)


class StyleContentLoss(nn.Module):
    """
    Объединённый лосс: контент + стиль + TV.
    """
    def __init__(
        self,
        content_weight: float = 1e5,
        style_weight: float = 1e10,
        tv_weight: float = 1e-6,
    ):
        super().__init__()
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight

    def forward(
        self,
        feats_out,
        feats_content,
        style_grams,
        generated: torch.Tensor,
    ):
        # feats_out, feats_content: список фичей VGG на разных слоях
        # style_grams: список Gram-матриц для стиля

        # Контент: возьмём 3-й уровень (relu3_3 / relu3_2 примерно)
        content_loss = F.mse_loss(feats_out[2], feats_content[2])

        # Стиль: суммируем по уровням
        style_loss = 0.0
        out_grams = [gram_matrix(f) for f in feats_out]
        for g_out, g_style in zip(out_grams, style_grams):
            style_loss += F.mse_loss(g_out, g_style)

        # TV loss (сглаживание)
        tv_loss = (
            torch.sum(torch.abs(generated[:, :, :, :-1] - generated[:, :, :, 1:])) +
            torch.sum(torch.abs(generated[:, :, :-1, :] - generated[:, :, 1:, :]))
        )

        total = (
            self.content_weight * content_loss +
            self.style_weight * style_loss +
            self.tv_weight * tv_loss
        )
        return total, content_loss, style_loss, tv_loss
