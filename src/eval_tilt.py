"""Контролируемая оценка точности крена на РЕАЛЬНЫХ фото грузовиков.

Метод (честный, задекларированный): берём чистые фото (детектируются как truck,
крен ~0°), поворачиваем на известные углы и ОБРЕЗАЕМ края (чтобы не появлялось
горизонтальных артефактов от замыливания границ). Это валидная синтетическая
аугментация — ground-truth = заданный угол. Сравниваем с измеренным.

Запуск:  python src/eval_tilt.py [dir=data/images/public] [max=40]
"""
from __future__ import annotations
import os
import sys
import glob
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import pipeline  # noqa: E402

ANGLES = [-12, -8, -5, -3, 3, 5, 8, 12]


def rotate_crop(img, deg, margin=0.15):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    rot = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    m = int(min(h, w) * margin)
    return rot[m:h - m, m:w - m]


def tilt_of(img):
    """|крен| и уверенность через боевой пайплайн (бокс прицепа + геометрия)."""
    r, _ = pipeline.analyze(img)
    if r.decision in ("NO-DATA", "CANNOT-ASSESS"):
        return None
    return r.tilt_deg


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "data/images/public"
    mx = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    files = sorted(glob.glob(os.path.join(d, "*.jpg")))[:mx]

    # берём «чистые» базовые фото: трак найден и крен ~0
    bases = []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        t = tilt_of(img)
        if t is not None and abs(t) < 2.0:
            bases.append((f, img, t))
    print(f"Чистых базовых фото: {len(bases)}  |  углы: {ANGLES}")

    per = {a: [] for a in ANGLES}
    errs = []
    for f, img, base in bases:
        for a in ANGLES:
            m = tilt_of(rotate_crop(img, a))
            if m is None:
                continue
            # поворот изображения на +a (CCW) -> измеренный угол ~ -a (отн. базы)
            err = abs((m - base) - (-a))
            per[a].append(err)
            errs.append(err)

    if not errs:
        print("нет данных")
        return
    print(f"\nTILT MAE: {np.mean(errs):.2f}°  |  median: {np.median(errs):.2f}°  |  n={len(errs)}")
    for a in ANGLES:
        v = per[a]
        if v:
            print(f"  {a:+d}°:  MAE {np.mean(v):.2f}°  (n={len(v)})")


if __name__ == "__main__":
    main()
