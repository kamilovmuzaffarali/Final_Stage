"""Юнит-тест ядра геометрии крена (без YOLO/torch — только OpenCV).
Рисуем кузов под известным углом и проверяем, что estimate_tilt близок к нему.
"""
import os
import sys
import math
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import estimate_tilt  # noqa: E402


def _draw_tilted_box(angle_deg: float, size=(400, 300)):
    img = np.full((size[1], size[0], 3), 230, np.uint8)
    cx, cy = size[0] // 2, size[1] // 2
    w, h = 300, 120
    rect = cv2.boxPoints(((cx, cy), (w, h), angle_deg)).astype(np.int32)
    cv2.drawContours(img, [rect], 0, (40, 40, 40), 3)
    # горизонтальные рёбра внутри (как борта кузова)
    for dy in (-30, 0, 30):
        p1 = (cx - w // 2, cy + dy)
        p2 = (cx + w // 2, cy + dy)
        M = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
        pts = np.array([[*p1, 1], [*p2, 1]]).T
        r = (M @ pts).T.astype(int)
        cv2.line(img, tuple(r[0]), tuple(r[1]), (40, 40, 40), 2)
    return img


def test_tilt_detection():
    for truth in (-8, -4, 0, 5, 9):
        img = _draw_tilted_box(truth)
        est = estimate_tilt(img)
        assert abs(est - truth) <= 3.0, f"angle {truth}: got {est}"


if __name__ == "__main__":
    for truth in (-8, -4, 0, 5, 9):
        est = estimate_tilt(_draw_tilted_box(truth))
        print(f"truth={truth:+d}  est={est:+.1f}  err={abs(est-truth):.1f}")
    print("OK")
