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
TILT_CONF_MIN = 0.45     # ниже -> «не оценить» (плохой ракурс/качество)
OVERHANG_WARN = 0.08     # доля вылета груза за габарит кузова (ширина)

# Пороги можно переопределить в config.yaml (без правки кода)
try:
    import os as _os
    import yaml as _yaml
    _cfg_path = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "config.yaml")
    if _os.path.exists(_cfg_path):
        with open(_cfg_path) as _f:
            _c = _yaml.safe_load(_f) or {}
        _t = _c.get("tilt", {})
        TILT_WARN_DEG = float(_t.get("warn_deg", TILT_WARN_DEG))
        TILT_NOGO_DEG = float(_t.get("nogo_deg", TILT_NOGO_DEG))
        TILT_CONF_MIN = float(_t.get("conf_min", TILT_CONF_MIN))
except Exception:
    pass

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


_world = None


def _trailer_box(image_bgr):
    """Тугой бокс кузова/прицепа через YOLO-World (lazy). None если не найден уверенно."""
    global _world
    try:
        if _world is None:
            from ultralytics import YOLO
            _world = YOLO("yolov8s-world.pt")
            _world.set_classes(["truck trailer", "truck cargo bed",
                                "dump truck bed", "box truck body"])
        r = _world.predict(image_bgr, verbose=False, conf=0.25)[0]
        if r.boxes is None or not len(r.boxes):
            return None
        confs = r.boxes.conf.cpu().numpy()
        xyxy = r.boxes.xyxy.cpu().numpy()
        i = int(confs.argmax())
        if confs[i] < 0.35:
            return None
        return tuple(xyxy[i].astype(int))
    except Exception:
        return None


def estimate_tilt(crop: np.ndarray) -> float:
    """Оценка крена кузова: угол доминантных ~горизонтальных линий кузова
    относительно горизонта (Canny + HoughLinesP + медиана углов)."""
    if crop.size == 0:
        return 0.0, 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    # длинные почти-горизонтальные линии кузова (борта/крыша/низ)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=int(crop.shape[1] * 0.5), maxLineGap=25)
    if lines is None:
        return 0.0, 0.0
    midy = crop.shape[0] / 2
    upper, lower = [], []           # верхняя и нижняя половина кузова
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(ang) >= 25:
            continue
        L = math.hypot(x2 - x1, y2 - y1)
        (upper if (y1 + y2) / 2 < midy else lower).append((ang, L))

    def _dom(cands):
        if not cands:
            return None
        cands.sort(key=lambda c: -c[1])
        top = cands[:max(2, len(cands) // 3)]
        w = sum(L for _, L in top)
        return sum(a * L for a, L in top) / w

    au, al = _dom(upper), _dom(lower)
    # Уверенность = наличие длинных кромок борта (структуры). Обе кромки -> высокая,
    # одна -> средняя, нет -> «не оценить». Усреднение верх+низ гасит перспективу.
    if au is not None and al is not None:
        return float((au + al) / 2), 1.0
    if au is not None or al is not None:
        return float(au if au is not None else al), 0.6
    return 0.0, 0.0


def analyze(image_bgr: np.ndarray) -> tuple[Result, np.ndarray]:
    """Главная функция: принимает BGR-изображение, возвращает (Result, annotated)."""
    # privacy-стадия: лица/номера пикселизуются ДО обработки и показа (правила)
    try:
        from src import privacy
        image_bgr, _ = privacy.sanitize(image_bgr)
    except Exception:
        pass
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

    # для крена берём ТУГОЙ бокс прицепа (YOLO-World) — без кабины/фона/перспективы;
    # если прицеп не найден, падаем на бокс грузовика (YOLOv8n)
    tbox = _trailer_box(image_bgr)
    x1, y1, x2, y2 = tbox if tbox is not None else box
    crop = image_bgr[max(0, y1):y2, max(0, x1):x2]
    tilt, conf = estimate_tilt(crop)

    # гейт уверенности: на косом ракурсе/плохом кадре не угадываем, а честно
    # отвечаем «не оценить» (как требует план: cannot assess)
    if conf < TILT_CONF_MIN:
        checks.append(Check("Крен кузова", True,
                            "не оценить (ракурс/качество)", "cannot"))
        cannot_roll = True
    else:
        cannot_roll = False
        a = abs(tilt)
        if a >= TILT_NOGO_DEG:
            checks.append(Check("Крен кузова", False, f"{tilt:+.1f}°", "nogo"))
        elif a >= TILT_WARN_DEG:
            checks.append(Check("Крен кузова", False, f"{tilt:+.1f}°", "warn"))
        else:
            checks.append(Check("Крен кузова", True, f"{tilt:+.1f}°", "ok"))

    # Свес за габарит / высота — честно помечены как ROADMAP и НЕ влияют на
    # решение. Zero-shot на грузе/колёсах ненадёжен (замерено) -> не выдаём
    # фейковый «OK». Включим после обучения своих классов cargo/wheel (train/).
    checks.append(Check("Свес за габарит", True, "не оценивается (roadmap)", "roadmap"))
    checks.append(Check("Высота груза", True, "не оценивается (roadmap)", "roadmap"))

    # --- итоговое решение ---
    note = ""
    if cannot_roll:
        decision = "CANNOT-ASSESS"
        color = (160, 160, 160)
        note = "Крен не оценить: снимите фуру сзади/сбоку, кузов целиком в кадре."
    else:
        sev = [c.severity for c in checks if c.severity not in ("roadmap", "cannot")]
        if "nogo" in sev:
            decision, color = "NO-GO", (0, 0, 255)
        elif "warn" in sev:
            decision, color = "WARN", (0, 165, 255)
        else:
            decision, color = "GO", (0, 180, 0)

    # аннотация
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
    label = decision if cannot_roll else f"{decision}  tilt={tilt:+.1f}"
    cv2.putText(annotated, label, (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    if not cannot_roll:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        L = (x2 - x1) // 2
        dx = int(L * math.cos(math.radians(tilt)))
        dy = int(L * math.sin(math.radians(tilt)))
        cv2.line(annotated, (cx - dx, cy - dy), (cx + dx, cy + dy), color, 2)

    result = Result(decision=decision, tilt_deg=tilt, checks=checks,
                    truck_found=True, note=note)
    return result, annotated


if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1])
    r, ann = analyze(img)
    print(r.to_dict())
    cv2.imwrite("annotated.jpg", ann)
    print("saved annotated.jpg")
