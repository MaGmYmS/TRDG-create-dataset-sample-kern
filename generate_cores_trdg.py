"""
generate_cores_trdg.py
Генерация изображений керна 250x130 с синтетическими рукописными надписями
и формированием датасета для обучения EasyOCR
"""

import os
import random
import argparse
from glob import glob

from trdg.generators import GeneratorFromStrings
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from tqdm import tqdm
import csv


# -----------------------
# Маски кодов
# -----------------------
MASK_PATTERNS = [
    "####_##.#п @@@",
    "###-###-###",
    "@@###.#####",
    "##_##_##_##",
    "###.п@###",
    "####-##-####",
    "@@###@@@",
    "##_@_##_@",
    "###-###.п",
    "####@@@@##",
]

LETTERS = "АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЭЮЯ"
DIGITS = "0123456789"


def generate_code_from_mask(mask: str) -> str:
    res = []
    for ch in mask:
        if ch == "#":
            res.append(random.choice(DIGITS))
        elif ch == "@":
            res.append(random.choice(LETTERS))
        else:
            res.append(ch)
    return "".join(res)


# -----------------------
# Преобразование текста TRDG в RGBA
# -----------------------
def trdg_image_to_rgba(img_pil, text_color=(20, 20, 20)):
    gray = img_pil.convert("L")
    arr = np.array(gray).astype(np.uint8)
    thresh = 230
    mask = np.where(arr < thresh, 255, 0).astype(np.uint8)

    mask_img = Image.fromarray(mask).filter(
        ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0))
    )

    color_layer = Image.new("RGBA", img_pil.size, text_color + (0,))
    solid = Image.new("RGBA", img_pil.size, text_color + (255,))
    color_layer.paste(solid, (0, 0), mask_img)
    return color_layer


# -----------------------
# Вставка текста в центр
# -----------------------
def paste_text_on_background(bg_pil, text_rgba):
    bg = bg_pil.copy().convert("RGBA")
    W, H = bg.size
    tx, ty = text_rgba.size

    scale = random.uniform(0.7, 0.99)
    target_w = int(W * scale)
    sf = target_w / tx
    target_h = int(ty * sf)
    text_resized = text_rgba.resize((target_w, target_h), resample=Image.LANCZOS)

    paste_x = W // 2 - target_w // 2
    paste_y = H // 2 - target_h // 2

    bg.paste(text_resized, (paste_x, paste_y), text_resized)
    return bg


# -----------------------
# Финальные аугментации
# -----------------------
def final_augment(img_pil):
    img = img_pil.copy()
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.9, 1.15))
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.95, 1.06))

    if random.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 0.8)))

    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, random.uniform(0.0, 6.0), arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# -----------------------
# Main
# -----------------------
def main(args):
    # создаём папки train и val
    train_dir = os.path.join(args.output_root, "ru_train_filtered")
    val_dir = os.path.join(args.output_root, "ru_val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    fonts = sorted(glob(os.path.join(args.fonts_dir, "*.ttf")))
    if not fonts:
        raise FileNotFoundError("Не найдено .ttf шрифтов в папке fonts/")

    bg_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        bg_paths.extend(glob(os.path.join(args.backgrounds_dir, ext)))
    if not bg_paths:
        raise FileNotFoundError("Не найдено фонов в папке background/")

    # открываем CSV файлы
    train_csv = open(os.path.join(train_dir, "labels.csv"), "w", newline="", encoding="utf-8")
    val_csv = open(os.path.join(val_dir, "labels.csv"), "w", newline="", encoding="utf-8")

    train_writer = csv.writer(train_csv)
    val_writer = csv.writer(val_csv)
    train_writer.writerow(["filename", "words"])
    val_writer.writerow(["filename", "words"])

    for i in tqdm(range(args.count)):
        label = generate_code_from_mask(random.choice(MASK_PATTERNS))

        generator = GeneratorFromStrings(
            [label],
            count=1,
            fonts=[random.choice(fonts)],
            size=args.text_size,
            skewing_angle=3,
            random_skew=True,
            blur=1,
            random_blur=True,
            background_type=1,
            is_handwritten=False,
            fit=True,
            margins=(0, 0, 0, 0),
        )

        trdg_img, _ = next(generator)
        text_rgba = trdg_image_to_rgba(trdg_img, text_color=(10, 10, 10))

        bg_path = random.choice(bg_paths)
        bg = Image.open(bg_path).convert("RGBA")

        # случайный поворот 0–180°
        angle = random.uniform(0, 180)
        bg = bg.rotate(angle, expand=True, resample=Image.BICUBIC)

        # вырезаем центральный кусок 250x130
        crop_w, crop_h = 250, 130
        W, H = bg.size
        left = max(0, W // 2 - crop_w // 2)
        top = max(0, H // 2 - crop_h // 2)
        right = left + crop_w
        bottom = top + crop_h
        bg = bg.crop((left, top, right, bottom))

        composite = paste_text_on_background(bg, text_rgba)
        out = final_augment(composite)
        # преобразуем в RGB перед сохранением как JPEG
        out = out.convert("RGB")

        # выбор train/val
        if random.random() < args.val_split:
            out_dir = val_dir
            writer = val_writer
        else:
            out_dir = train_dir
            writer = train_writer

        fname = f"{i:06d}__{label}.jpg"
        fpath = os.path.join(out_dir, fname)
        out.save(fpath, quality=95)

        # пишем в CSV
        writer.writerow([fname, label])

    train_csv.close()
    val_csv.close()
    print("Готово. Датасет сохранён в", args.output_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20, help="Общее число изображений")
    parser.add_argument("--fonts_dir", type=str, default="fonts")
    parser.add_argument("--backgrounds_dir", type=str, default="background")
    parser.add_argument("--output_root", type=str, default="all_data")
    parser.add_argument("--text_size", type=int, default=64, help="Размер текста TRDG")
    parser.add_argument("--val_split", type=float, default=0.1, help="Доля выборки для валидации")
    args = parser.parse_args()
    main(args)
