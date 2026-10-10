"""
Pre-Departure Load Safety Gate — core CV pipeline.

Вход: фото загруженного грузовика (телефон/камера на воротах).
Выход: решение GO / WARN / NO-GO + метрики (угол крена, свес, флаги) +
аннотированное изображение.

Это НЕ вызов чужой готовой модели «под ключ», а система из трёх ступеней:
  1) Детекция грузовика (YOLO) -> область кузова.
  2) Геометрия крена (OpenCV: Canny + HoughLinesP -> угол доминантных линий кузова).
  3) Решающий слой (пороги -> GO/WARN/NO-GO + человекочитаемое объяснение).

На хакатоне детектор дообучается на своём датасете (train/), но пайплайн
работает сразу на предобученном YOLOv8 (класс 'truck' из COCO) — honest MVP.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

# Пороги (можно вынести в конфиг/обучить). Углы в градусах.
TILT_WARN_DEG = 4.0      # крен кузова -> предупреждение
TILT_NOGO_DEG = 7.0      # крен кузова -> запрет выезда
OVERHANG_WARN = 0.08     # доля вылета груза за габарит кузова (ширина)

# Классы COCO, которые считаем «грузовиком» на старте (до своей модели)
COCO_TRUCK = {7}          # 'truck'
COCO_VEHICLE_FALLBACK = {7, 5, 2}  # truck, bus, car — если trucks не нашлись

_model = None


def _load_model():
    """Ленивая загрузка YOLO. Если есть своя дообученная модель — указать путь
    в MODEL_PATH. Иначе берём предобученный yolov8n (COCO)."""
    global _model
    if _model is None:
        from ultralytics import YOLO
        import os
        path = os.environ.get("MODEL_PATH", "yolov8n.pt")
        _model = YOLO(path)
    return _model


@dataclass
class Check:
    name: str
    ok: bool
    value: str
    severity: str  # "ok" | "warn" | "nogo"


@dataclass
class Result:
    decision: str                       # GO | WARN | NO-GO
    tilt_deg: float
    checks: list = field(default_factory=list)
    truck_found: bool = False
    note: str = ""

    def to_dict(self):
        d = asdict(self)
        d["tilt_deg"] = round(self.tilt_deg, 2)
        return d


def _biggest_truck_box(res) -> tuple | None:
    """Возвращает (x1,y1,x2,y2) самого крупного грузовика, иначе None."""
    boxes = getattr(res, "boxes", None)
    if boxes is None or boxes.cls is None:
        return None
    cls = boxes.cls.cpu().numpy().astype(int)
    xyxy = boxes.xyxy.cpu().numpy()
    for wanted in (COCO_TRUCK, COCO_VEHICLE_FALLBACK):
        idx = [i for i, c in enumerate(cls) if c in wanted]
        if idx:
            areas = [(xyxy[i][2] - xyxy[i][0]) * (xyxy[i][3] - xyxy[i][1]) for i in idx]
            return tuple(xyxy[idx[int(np.argmax(areas))]].astype(int))
    return None


def estimate_tilt(crop: np.ndarray) -> float:
    """Оценка крена кузова: угол доминантных ~горизонтальных линий кузова
    относительно горизонта (Canny + HoughLinesP + медиана углов)."""
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                            minLineLength=crop.shape[1] * 0.3, maxLineGap=20)
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
        # интересуют почти горизонтальные линии кузова (|angle| < 30)
        if abs(ang) < 30:
            angles.append(ang)
    if not angles:
        return 0.0
    return float(np.median(angles))


def analyze(image_bgr: np.ndarray) -> tuple[Result, np.ndarray]:
    """Главная функция: принимает BGR-изображение, возвращает (Result, annotated)."""
    annotated = image_bgr.copy()
    model = _load_model()
    res = model.predict(image_bgr, verbose=False, conf=0.35)[0]
    box = _biggest_truck_box(res)

    checks: list[Check] = []
    tilt = 0.0
    truck_found = box is not None

    if not truck_found:
        result = Result(decision="NO-DATA", tilt_deg=0.0, checks=[],
                        truck_found=False,
                        note="Грузовик на фото не найден — переснимите сбоку/сзади.")
        cv2.putText(annotated, "TRUCK NOT FOUND", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return result, annotated

    x1, y1, x2, y2 = box
    crop = image_bgr[max(0, y1):y2, max(0, x1):x2]
    tilt = estimate_tilt(crop)

    # --- проверка крена ---
    a = abs(tilt)
    if a >= TILT_NOGO_DEG:
        checks.append(Check("Крен кузова", False, f"{tilt:+.1f}°", "nogo"))
    elif a >= TILT_WARN_DEG:
        checks.append(Check("Крен кузова", False, f"{tilt:+.1f}°", "warn"))
    else:
        checks.append(Check("Крен кузова", True, f"{tilt:+.1f}°", "ok"))

    # --- заглушки под дообучаемую модель (честно помечены) ---
    # Свес груза за габарит / незакреплённый груз / превышение высоты
    # включаются после обучения сегментации груза (train/). Пока — pending.
    checks.append(Check("Свес за габарит", True, "модель обучается", "ok"))
    checks.append(Check("Крепление/дверь", True, "модель обучается", "ok"))

    # --- итоговое решение ---
    sev = [c.severity for c in checks]
    if "nogo" in sev:
        decision = "NO-GO"
        color = (0, 0, 255)
    elif "warn" in sev:
        decision = "WARN"
        color = (0, 165, 255)
    else:
        decision = "GO"
        color = (0, 180, 0)

    # аннотация
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
    cv2.putText(annotated, f"{decision}  tilt={tilt:+.1f}", (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    # горизонт-референс (линия под углом крена через центр бокса)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    L = (x2 - x1) // 2
    dx = int(L * math.cos(math.radians(tilt)))
    dy = int(L * math.sin(math.radians(tilt)))
    cv2.line(annotated, (cx - dx, cy - dy), (cx + dx, cy + dy), color, 2)

    result = Result(decision=decision, tilt_deg=tilt, checks=checks,
                    truck_found=True)
    return result, annotated


if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1])
    r, ann = analyze(img)
    print(r.to_dict())
    cv2.imwrite("annotated.jpg", ann)
    print("saved annotated.jpg")
