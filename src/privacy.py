"""Privacy-стадия: детектим лица и номерные знаки и пикселизуем их ДО обработки/показа.
Соответствие правилам: «no faces or plates of people who did not consent».

Используем Haar-каскады, встроенные в OpenCV (без внешних загрузок):
  - haarcascade_frontalface_default.xml  (лица)
  - haarcascade_russian_plate_number.xml (номерные знаки)
"""
from __future__ import annotations
import os
import cv2
import numpy as np

_CASC = cv2.data.haarcascades
_face = cv2.CascadeClassifier(os.path.join(_CASC, "haarcascade_frontalface_default.xml"))
_plate = cv2.CascadeClassifier(os.path.join(_CASC, "haarcascade_russian_plate_number.xml"))


def _pixelate(img, x, y, w, h, blocks=8):
    x2, y2 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
    x, y = max(x, 0), max(y, 0)
    roi = img[y:y2, x:x2]
    if roi.size == 0:
        return
    small = cv2.resize(roi, (max(1, blocks), max(1, blocks)),
                       interpolation=cv2.INTER_LINEAR)
    img[y:y2, x:x2] = cv2.resize(small, (x2 - x, y2 - y),
                                 interpolation=cv2.INTER_NEAREST)


def sanitize(img_bgr: np.ndarray) -> tuple[np.ndarray, int]:
    """Возвращает (изображение с заблюренными лицами/номерами, число скрытых областей)."""
    out = img_bgr.copy()
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    n = 0
    for casc, scale, nb, mins in ((_face, 1.1, 5, (30, 30)),
                                  (_plate, 1.1, 4, (40, 15))):
        try:
            dets = casc.detectMultiScale(gray, scale, nb, minSize=mins)
        except Exception:
            dets = []
        for (x, y, w, h) in dets:
            _pixelate(out, x, y, w, h)
            n += 1
    return out, n


if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1])
    clean, n = sanitize(img)
    cv2.imwrite("sanitized.jpg", clean)
    print(f"скрыто областей (лица/номера): {n} -> sanitized.jpg")
