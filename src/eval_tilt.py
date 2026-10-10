"""Оценка точности измерения крена на РЕАЛЬНЫХ фото грузовиков.

Метод (контролируемый тест-сет): берём реальные фото, поворачиваем на известные
углы (ground-truth), прогоняем пайплайн и сравниваем измеренный крен с заданным.
Выдаёт MAE (°) общий и по углам -> для раздела «метрики» в деке.

Запуск:  python src/eval_tilt.py [dir=data/images/public] [max=20]
"""
from __future__ import annotations
import os
import sys
import glob
import math
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import analyze, estimate_tilt, _load_model, _biggest_truck_box  # noqa

ANGLES = [-10, -6, -3, 0, 3, 6, 10]


def rotate(img, deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def measure(img):
    """Крен на реальном фото: детектим грузовик, меряем угол кузова."""
    model = _load_model()
    res = model.predict(img, verbose=False, conf=0.35)[0]
    box = _biggest_truck_box(res)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    crop = img[max(0, y1):y2, max(0, x1):x2]
    t,_=estimate_tilt(crop); return t


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "data/images/public"
    mx = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    files = sorted(glob.glob(os.path.join(d, "*.jpg")))[:mx]
    print(f"Фото: {len(files)} | углы: {ANGLES}")

    per_angle = {a: [] for a in ANGLES}
    base0 = {}  # базовый крен каждого фото (при 0°) — вычитаем
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        b = measure(img)
        base0[f] = b if b is not None else None

    errs = []
    for f in files:
        img = cv2.imread(f)
        if img is None or base0.get(f) is None:
            continue
        for a in ANGLES:
            m = measure(rotate(img, a))
            if m is None:
                continue
            # поворот на +a по часовой -> измеренный крен меняется на -a отн. базы
            pred_delta = m - base0[f]
            err = abs(pred_delta - (-a))
            per_angle[a].append(err)
            errs.append(err)

    if not errs:
        print("нет данных")
        return
    print(f"\nTILT MAE (общий): {np.mean(errs):.2f}°  |  медиана: {np.median(errs):.2f}°  |  n={len(errs)}")
    print("по углам:")
    for a in ANGLES:
        v = per_angle[a]
        if v:
            print(f"  {a:+d}°:  MAE {np.mean(v):.2f}°  (n={len(v)})")


if __name__ == "__main__":
    main()
