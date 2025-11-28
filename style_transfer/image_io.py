# style_transfer/image_io.py

import os
from typing import List, Optional

from PIL import Image
import torch
from torchvision import transforms as T


def build_preprocess(max_size: Optional[int] = None):
    """
    Преобразование PIL.Image -> тензор [C,H,W] в диапазоне [0, 255].
    При max_size картинка масштабируется так, чтобы большая сторона <= max_size.
    """
    transforms_list = []

    if max_size is not None:
        transforms_list.append(T.Resize(max_size))

    transforms_list.extend([
        T.ToTensor(),                      # [0, 1]
        T.Lambda(lambda x: x * 255.0)      # [0, 255]
    ])

    return T.Compose(transforms_list)


def tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """
    Тензор [C,H,W] или [1,C,H,W] -> PIL.Image (uint8).
    """
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]

    img = img_tensor.detach().cpu().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")  # C,H,W -> H,W,C
    return Image.fromarray(img)


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_image(img: Image.Image, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def list_images_in_dir(folder: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    result = []
    for name in os.listdir(folder):
        ext = os.path.splitext(name)[1].lower()
        if ext in exts:
            result.append(os.path.join(folder, name))
    result.sort()
    return result
