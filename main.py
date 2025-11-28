# main.py

import os
import argparse

import torch

from style_transfer import StyleTransferNet, image_io


def stylize_single(
    model: StyleTransferNet,
    content_path: str,
    output_path: str,
    preprocess,
    device: torch.device,
):
    # Загрузка исходного изображения
    img = image_io.load_image(content_path)
    tensor = preprocess(img).unsqueeze(0).to(device)  # [1,3,H,W]

    with torch.no_grad():
        out_tensor = model(tensor)

    out_img = image_io.tensor_to_pil(out_tensor)
    image_io.save_image(out_img, output_path)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Перенос художественного стиля на фотографии (только инференс)."
    )
    parser.add_argument(
        "--style-checkpoint",
        required=True,
        help="Путь к .pth файлу с весами стилевой модели.",
    )
    parser.add_argument(
        "--content",
        help="Путь к одной фотографии (например, images/photo.jpg).",
    )
    parser.add_argument(
        "--content-dir",
        help="Папка с фотографиями (например, images/photos/).",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Папка для сохранения результата (по умолчанию: ./output).",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1024,
        help="Максимальный размер по большей стороне (по умолчанию 1024).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help='Устройство: "auto" (по умолчанию), "cpu" или "cuda". '
             'При "cuda" без доступного GPU будет ошибка.',
    )

    args = parser.parse_args()

    if not args.content and not args.content_dir:
        raise SystemExit("Нужно указать либо --content, либо --content-dir")

    # Определяем устройство
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Используется устройство: {device}")

    # Создаём модель и загружаем веса
    print(f"Загружаю веса стиля: {args.style_checkpoint}")
    checkpoint = torch.load(args.style_checkpoint, map_location=device)

    model = StyleTransferNet()
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    preprocess = image_io.build_preprocess(max_size=args.max_size)

    # Одна картинка
    if args.content:
        name = os.path.basename(args.content)
        out_path = os.path.join(args.output_dir, name)
        stylize_single(model, args.content, out_path, preprocess, device)

    # Папка с картинками
    if args.content_dir:
        paths = image_io.list_images_in_dir(args.content_dir)
        if not paths:
            print(f"В папке {args.content_dir} не найдено картинок.")
        for path in paths:
            name = os.path.basename(path)
            out_path = os.path.join(args.output_dir, name)
            stylize_single(model, path, out_path, preprocess, device)


if __name__ == "__main__":
    main()
