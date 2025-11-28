import os
import argparse
import csv

import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image

from style_transfer import StyleTransferNet
from style_transfer.vgg import VGGFeatures, vgg_normalize_batch
from style_transfer.losses import StyleContentLoss, gram_matrix

import torch.nn.functional as F
from pytorch_msssim import ssim
import lpips

import matplotlib.pyplot as plt


class FlatImageDataset(Dataset):
    """
    Датасет, который просто собирает все картинки из папки (и подпапок).
    Никаких классов, только сами изображения.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.paths = []
        for r, _, files in os.walk(root_dir):
            for name in files:
                if os.path.splitext(name)[1].lower() in exts:
                    self.paths.append(os.path.join(r, name))
        if not self.paths:
            raise FileNotFoundError(f"Не найдено ни одной картинки в {root_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # второй элемент (label) нам не нужен, вернём 0
        return img, 0


def make_dataloader(dataset_dir: str, image_size: int, batch_size: int):
    transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Lambda(to_255),  # [0,255]
    ])
    dataset = FlatImageDataset(dataset_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader


def to_255(x):
    return x * 255.0


def prepare_style_grams(style_path: str, device, vgg: VGGFeatures, image_size: int):
    img = Image.open(style_path).convert("RGB")
    transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Lambda(to_255),
    ])
    style = transform(img).unsqueeze(0).to(device)  # [1,3,H,W]
    style_norm = vgg_normalize_batch(style)
    with torch.no_grad():
        feats_style = vgg(style_norm)
    style_grams = [gram_matrix(f) for f in feats_style]
    return style_grams, style


# ---- ФУНКЦИЯ ДЛЯ ЦВЕТОВОЙ ДИСТАНЦИИ ----
def color_histogram_distance(img: torch.Tensor,
                             ref: torch.Tensor,
                             num_bins: int = 32) -> torch.Tensor:
    """
    Простая метрика различия цветовых гистограмм между img и ref.
    img, ref: [B,3,H,W], диапазон [0,255]
    Используем Bhattacharyya distance по каждому каналу и усредняем.
    """
    img = img[0].detach().cpu().view(3, -1)
    ref = ref[0].detach().cpu().view(3, -1)

    dist = 0.0
    for c in range(3):
        h1 = torch.histc(img[c], bins=num_bins, min=0.0, max=255.0)
        h2 = torch.histc(ref[c], bins=num_bins, min=0.0, max=255.0)
        h1 = h1 / (h1.sum() + 1e-8)
        h2 = h2 / (h2.sum() + 1e-8)
        # Bhattacharyya coefficient
        bc = torch.sum(torch.sqrt(h1 * h2))
        # Bhattacharyya distance
        d = torch.sqrt(torch.clamp(1.0 - bc, min=0.0))
        dist += d
    return dist / 3.0


def main():
    parser = argparse.ArgumentParser(
        description="Обучение сети переноса стиля для фотографий."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Путь к датасету изображений (корень для ImageFolder, например dataset/train2014).",
    )
    parser.add_argument(
        "--style-image",
        required=True,
        help="Путь к изображению стиля (например, images/style/starry_night.jpg).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Размер картинок при обучении (по умолчанию 256).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Размер батча (по умолчанию 4).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Количество эпох (по умолчанию 2).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (по умолчанию 1e-3).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Куда сохранять веса модели (по умолчанию ./checkpoints).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help='Устройство: "auto", "cpu" или "cuda".',
    )

    args = parser.parse_args()

    # Устройство
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Используется устройство: {device}")

    # Датасет
    loader = make_dataloader(args.dataset, args.image_size, args.batch_size)
    print(f"Обучающих изображений: {len(loader.dataset)}")

    # Модель и оптимизатор
    model = StyleTransferNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # VGG для лоссов
    vgg = VGGFeatures().to(device).eval()

    # Стиль (Gram-матрицы + изображение для метрик)
    style_grams, style_img_for_metrics = prepare_style_grams(
        args.style_image, device, vgg, args.image_size
    )

    # LPIPS (перцептивная дистанция)
    lpips_fn = lpips.LPIPS(net='alex').to(device).eval()

    # Лосс
    criterion = StyleContentLoss(
        content_weight=7.5e4,
        style_weight=2e10,
        tv_weight=5e-6,
    )

    # История метрик по эпохам
    epoch_metrics = []

    # Обучение
    model.train()
    for epoch in range(1, args.epochs + 1):
        running_total = 0.0
        running_content = 0.0
        running_style_loss = 0.0
        running_tv = 0.0

        # метрики
        running_ssim = 0.0
        running_lpips_content = 0.0
        running_lpips_style = 0.0
        running_color_dist = 0.0

        for i, (batch, _) in enumerate(loader, start=1):
            batch = batch.to(device)  # [B,3,H,W]

            # Прямой проход через модель
            out = model(batch)

            # Нормализация для VGG
            batch_norm = vgg_normalize_batch(batch)
            out_norm = vgg_normalize_batch(out)

            # Фичи контента и выхода
            with torch.no_grad():
                feats_content = vgg(batch_norm)
            feats_out = vgg(out_norm)

            # Лоссы
            total_loss, c_loss, s_loss, tv_loss = criterion(
                feats_out, feats_content, style_grams, out
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # ---- МЕТРИКИ (без градиентов) ----
            with torch.no_grad():
                # Приводим к [0,1] для SSIM
                out_01 = out.clamp(0.0, 255.0) / 255.0
                batch_01 = batch.clamp(0.0, 255.0) / 255.0

                # SSIM (контент ↔ результат)
                ssim_val = ssim(out_01, batch_01, data_range=1.0)

                # LPIPS требует [-1,1]
                out_lpips = out_01 * 2.0 - 1.0
                batch_lpips = batch_01 * 2.0 - 1.0

                lpips_content = lpips_fn(out_lpips, batch_lpips).mean()

                # Стилевое изображение подгоняем по размеру к out
                style_resized = F.interpolate(
                    style_img_for_metrics,
                    size=out.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )

                style_01 = style_resized.clamp(0.0, 255.0) / 255.0
                style_lpips = style_01 * 2.0 - 1.0

                lpips_style = lpips_fn(out_lpips, style_lpips).mean()

                # Цветовая дистанция (гистограммы)
                color_dist = color_histogram_distance(out, style_resized)

            # Накопление
            n = i
            running_total += total_loss.item()
            running_content += c_loss.item()
            running_style_loss += s_loss.item()
            running_tv += tv_loss.item()

            running_ssim += ssim_val.item()
            running_lpips_content += lpips_content.item()
            running_lpips_style += lpips_style.item()
            running_color_dist += color_dist.item()

            if i % 50 == 0:
                print(
                    f"Epoch [{epoch}/{args.epochs}] Step [{i}/{len(loader)}] "
                    f"total: {running_total / n:.2f} "
                    f"content: {running_content / n:.2f} "
                    f"style: {running_style_loss / n:.2f} "
                    f"tv: {running_tv / n:.4f} "
                    f"ssim: {running_ssim / n:.4f} "
                    f"lpips_c: {running_lpips_content / n:.4f} "
                    f"lpips_s: {running_lpips_style / n:.4f} "
                    f"color_dist: {running_color_dist / n:.4f}"
                )

        # Метрики за эпоху (усреднённые)
        num_batches = len(loader)
        epoch_info = {
            "epoch": epoch,
            "total_loss": running_total / num_batches,
            "content_loss": running_content / num_batches,
            "style_loss": running_style_loss / num_batches,
            "tv_loss": running_tv / num_batches,
            "ssim": running_ssim / num_batches,
            "lpips_content": running_lpips_content / num_batches,
            "lpips_style": running_lpips_style / num_batches,
            "color_dist": running_color_dist / num_batches,
        }
        epoch_metrics.append(epoch_info)
        print("== Epoch summary:", epoch_info)

        # Сохранить checkpoint после каждой эпохи
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(
            args.checkpoint_dir,
            f"style_epoch_{epoch}.pth"
        )
        torch.save(model.state_dict(), ckpt_path)
        print(f"Сохранён чекпоинт: {ckpt_path}")


    # 1) Сохранение метрик в CSV
    metrics_csv_path = os.path.join(args.checkpoint_dir, "metrics.csv")
    if epoch_metrics:
        fieldnames = list(epoch_metrics[0].keys())
        with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in epoch_metrics:
                writer.writerow(row)
        print(f"Метрики по эпохам сохранены в {metrics_csv_path}")

    # 2) Построение графиков
    plots_dir = os.path.join(args.checkpoint_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    epochs = [m["epoch"] for m in epoch_metrics]

    def plot_metric(metric_name: str, ylabel: str):
        values = [m[metric_name] for m in epoch_metrics]
        plt.figure()
        plt.plot(epochs, values, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(metric_name)
        plt.grid(True)
        out_path = os.path.join(plots_dir, f"{metric_name}.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"Сохранён график {metric_name}: {out_path}")

    # Лоссы
    plot_metric("total_loss", "Total loss")
    plot_metric("content_loss", "Content loss")
    plot_metric("style_loss", "Style loss")
    plot_metric("tv_loss", "TV loss")

    # Метрики качества
    plot_metric("ssim", "SSIM (content vs output)")
    plot_metric("lpips_content", "LPIPS (content vs output)")
    plot_metric("lpips_style", "LPIPS (style vs output)")
    plot_metric("color_dist", "Color histogram distance (style vs output)")


if __name__ == "__main__":
    main()
