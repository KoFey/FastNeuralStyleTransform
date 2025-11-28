# style_transfer/vgg.py

import torch
from torch import nn
from torchvision.models import vgg16, VGG16_Weights


class VGGFeatures(nn.Module):
    """
    VGG16, обрезанная до нужных слоёв.
    Используется ТОЛЬКО для извлечения признаков (лоссы).
    """
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features

        # Индексы слоёв, по которым будем брать фичи
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(0, 4)])   # relu1_2
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(4, 9)])   # relu2_2
        self.slice3 = nn.Sequential(*[vgg[i] for i in range(9, 16)])  # relu3_3
        self.slice4 = nn.Sequential(*[vgg[i] for i in range(16, 23)]) # relu4_3

        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        """
        Возвращает список фичей на нескольких уровнях.
        """
        h = self.slice1(x)
        feat1 = h
        h = self.slice2(h)
        feat2 = h
        h = self.slice3(h)
        feat3 = h
        h = self.slice4(h)
        feat4 = h
        return [feat1, feat2, feat3, feat4]


def vgg_normalize_batch(batch: torch.Tensor) -> torch.Tensor:
    """
    Нормализация под VGG: ожидает вход в [0, 255].
    Приводим к [0,1], затем вычитаем mean и делим на std.
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=batch.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=batch.device).view(1, 3, 1, 1)
    x = batch / 255.0
    return (x - mean) / std
