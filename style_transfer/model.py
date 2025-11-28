# style_transfer/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Свёрточный блок: ReflectionPad2d -> Conv2d -> InstanceNorm2d -> ReLU (опционально).
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, use_relu=True):
        super().__init__()
        padding = kernel_size // 2
        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.use_relu = use_relu

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        if self.use_relu:
            x = F.relu(x, inplace=True)
        return x


class ResidualBlock(nn.Module):
    """
    Остаточный блок: x -> ConvBlock -> ConvBlock (без ReLU в конце) + skip connection.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size=3, stride=1, use_relu=True)
        self.conv2 = ConvBlock(channels, channels, kernel_size=3, stride=1, use_relu=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual


class UpsampleConvBlock(nn.Module):
    """
    Upsample (nearest) -> ReflectionPad2d -> Conv2d -> InstanceNorm2d -> ReLU (опц.).
    Такой вариант даёт меньше артефактов, чем ConvTranspose2d.
    """
    def __init__(self, in_ch, out_ch, kernel_size, scale_factor=2, use_relu=True):
        super().__init__()
        padding = kernel_size // 2
        self.scale_factor = scale_factor
        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.use_relu = use_relu

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        if self.use_relu:
            x = F.relu(x, inplace=True)
        return x


class StyleTransferNet(nn.Module):
    """
    Сеть переноса стиля для изображений (feed-forward).
    Архитектура:
      - 3 сверточных слоя (downsampling)
      - несколько residual-блоков
      - 2 upsample-блока
      - финальный conv 3->3 без нормализации и ReLU
    """
    def __init__(self, num_residuals: int = 5):
        super().__init__()

        # Downsampling
        self.conv1 = ConvBlock(3, 32, kernel_size=9, stride=1)
        self.conv2 = ConvBlock(32, 64, kernel_size=3, stride=2)
        self.conv3 = ConvBlock(64, 128, kernel_size=3, stride=2)

        # Residual blocks
        res_blocks = []
        for _ in range(num_residuals):
            res_blocks.append(ResidualBlock(128))
        self.residuals = nn.Sequential(*res_blocks)

        # Upsampling
        self.deconv1 = UpsampleConvBlock(128, 64, kernel_size=3, scale_factor=2)
        self.deconv2 = UpsampleConvBlock(64, 32, kernel_size=3, scale_factor=2)

        # Output layer (без нормализации, без ReLU)
        self.pad = nn.ReflectionPad2d(9 // 2)
        self.conv_out = nn.Conv2d(32, 3, kernel_size=9)

    def forward(self, x):
        # Ожидаем вход в диапазоне [0, 255]
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)

        y = self.residuals(y)

        y = self.deconv1(y)
        y = self.deconv2(y)

        y = self.pad(y)
        y = self.conv_out(y)

        # Выход тоже в [0, 255], обрежем на всякий случай
        return torch.clamp(y, 0.0, 255.0)
