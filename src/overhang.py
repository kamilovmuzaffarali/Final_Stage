"""Свес груза за габарит — второй чек. Zero-shot (YOLO-World), ленивая загрузка.

Идея: находим бокс КУЗОВА (борт/прицеп/платформа) и бокс ГРУЗА.
Боковой свес = насколько груз выходит за кузов по ширине, в долях ширины кузова.
Переводим в см по стандарту ширины кузова 2.55 м (config).

Детектор колёс/груза слабый на крытых фурах — поэтому честно: если боксы
не найдены уверенно, возвращаем «cannot assess», а не выдумываем число.
"""
from __future__ import annotations
import numpy as np

BODY_PROMPTS = ["truck trailer", "truck cargo bed", "open truck body",
                "flatbed truck bed", "dump truck bed"]
CARGO_PROMPTS = ["cargo on truck", "load on flatbed", "pile of soil on truck",
                 "stacked boxes on truck", "cargo load above truck sides"]
ALL = BODY_PROMPTS + CARGO_PROMPTS

BODY_WIDTH_M = 2.55          # стандартная ширина кузова -> масштаб
MIN_CONF_BODY = 0.25
MIN_CONF_CARGO = 0.25

_world = None


def _load():
    global _world
    if _world is None:
        from ultralytics import YOLO
        _world = YOLO("yolov8s-world.pt")
    return _world


def _best(dets, labels, min_conf):
    cand = [d for d in dets if d[0] in labels and d[1] >= min_conf]
    if not cand:
        return None
    return max(cand, key=lambda d: d[1])  # (label, conf, (x1,y1,x2,y2))


def assess(image_bgr):
    """Возвращает dict: {status, value, severity, ...}. status: ok|overhang|cannot."""
    model = _load()
    model.set_classes(ALL)
    r = model.predict(image_bgr, verbose=False, conf=0.04)[0]
    dets = []
    if r.boxes is not None:
        names = r.names
        for c, conf, xyxy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(),
                                 r.boxes.xyxy.tolist()):
            dets.append((names[int(c)], float(conf), [int(v) for v in xyxy]))

    body = _best(dets, BODY_PROMPTS, MIN_CONF_BODY)
    cargo = _best(dets, CARGO_PROMPTS, MIN_CONF_CARGO)
    if body is None or cargo is None:
        return {"status": "cannot", "value": "не оценить (нет кузова/груза)",
                "severity": "ok", "body": body, "cargo": cargo}

    bx1, by1, bx2, by2 = body[2]
    cx1, cy1, cx2, cy2 = cargo[2]
    bw = max(1, bx2 - bx1)
    left = max(0, bx1 - cx1)
    right = max(0, cx2 - bx2)
    frac = (left + right) / bw
    cm = frac * BODY_WIDTH_M * 100
    side = "слева" if left >= right else "справа"
    return {"status": "overhang" if frac > 0.03 else "ok",
            "value": f"~{cm:.0f} см {side}" if frac > 0.03 else "в габарите",
            "frac": round(frac, 3), "severity": "nogo" if frac > 0.08 else
            ("warn" if frac > 0.03 else "ok"),
            "body": body, "cargo": cargo}


if __name__ == "__main__":
    import sys, cv2
    img = cv2.imread(sys.argv[1])
    print(assess(img))
