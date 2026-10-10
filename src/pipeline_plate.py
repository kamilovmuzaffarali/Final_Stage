"""PlateVision — инференс-пайплайн (наша система, не вызов одной чужой модели):

  1) Детекция номера — наш YOLOv8n, обученный на наших композитах.
  2) Кроп + нормализация.
  3) Распознавание — наш CRNN (обучен на нашем синтетическом UZ-датасете).
  4) Проверка формата UZ (регэксп) + гейт уверенности -> «не читаю» вместо угадывания.

Выход: Result(text, decision, conf, box, valid) + аннотированное изображение.
"""
from __future__ import annotations
import os, re, math
from dataclasses import dataclass, asdict

import cv2
import numpy as np
import torch

from src.ocr_model import CRNN, ctc_greedy_decode, IDX2CH

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OCR_PATH = os.path.join(_ROOT, "weights", "ocr_crnn.pt")


def _find_det():
    import glob as _g
    cands = _g.glob(os.path.join(_ROOT, "runs", "**", "plate_det*", "weights", "best.pt"),
                    recursive=True)
    env = os.environ.get("DET_PATH")
    if env and os.path.exists(env):
        return env
    return max(cands, key=os.path.getmtime) if cands else os.path.join(
        _ROOT, "runs", "detect", "runs", "plate_det", "weights", "best.pt")


DET_PATH = _find_det()

# Формат UZ: "RR L DDD LL" или "RR DDD LLL"
UZ_RE = re.compile(r"^\d{2}([A-Z]\d{3}[A-Z]{2}|\d{3}[A-Z]{3})$")
OCR_CONF_MIN = 0.55
DET_CONF_MIN = 0.35

_det = None
_ocr = None
_dev = "mps" if torch.backends.mps.is_available() else "cpu"


def _load():
    global _det, _ocr
    if _ocr is None:
        _ocr = CRNN().to(_dev).eval()
        ck = torch.load(OCR_PATH, map_location=_dev)
        _ocr.load_state_dict(ck["model"])
    if _det is None:
        from ultralytics import YOLO
        _det = YOLO(_find_det())
    return _det, _ocr


@dataclass
class Result:
    decision: str            # READ | CANNOT-READ | NO-PLATE
    text: str
    valid_format: bool
    ocr_conf: float
    det_conf: float
    box: tuple | None
    note: str = ""

    def to_dict(self):
        d = asdict(self)
        d["ocr_conf"] = round(self.ocr_conf, 3)
        d["det_conf"] = round(self.det_conf, 3)
        return d


def _recognize(crop_bgr) -> tuple[str, float]:
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (128, 32)).astype(np.float32)
    g = (g / 255.0 - 0.5) / 0.5
    x = torch.from_numpy(g).unsqueeze(0).unsqueeze(0).to(_dev)
    with torch.no_grad():
        logits = _ocr(x)                      # (T,1,C)
    probs = logits.softmax(2).cpu()
    text = ctc_greedy_decode(logits.cpu())[0]
    # уверенность: средний max-prob по шагам, где выбран не-blank
    idx = probs.argmax(2).squeeze(1)
    maxp = probs.max(2).values.squeeze(1)
    nb = maxp[idx != 0]
    conf = float(nb.mean()) if len(nb) else 0.0
    return text, conf


def analyze(image_bgr) -> tuple[Result, np.ndarray]:
    det, _ = _load()
    annotated = image_bgr.copy()
    r = det.predict(image_bgr, verbose=False, conf=DET_CONF_MIN)[0]
    if r.boxes is None or not len(r.boxes):
        cv2.putText(annotated, "NO PLATE", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return Result("NO-PLATE", "", False, 0.0, 0.0, None,
                      "Номер не найден — снимите ближе/прямее."), annotated

    confs = r.boxes.conf.cpu().numpy()
    i = int(confs.argmax())
    det_conf = float(confs[i])
    x1, y1, x2, y2 = r.boxes.xyxy.cpu().numpy()[i].astype(int)
    crop = image_bgr[max(0, y1):y2, max(0, x1):x2]
    text, ocr_conf = _recognize(crop)
    valid = bool(UZ_RE.match(text))

    if ocr_conf < OCR_CONF_MIN or not valid:
        decision = "CANNOT-READ"
        color = (160, 160, 160)
        note = "Низкая уверенность/формат не UZ — переснимите прямее, крупнее."
    else:
        decision = "READ"
        color = (0, 180, 0)
        note = ""

    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
    label = text if text else "?"
    cv2.putText(annotated, label, (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return Result(decision, text, valid, ocr_conf, det_conf,
                  (int(x1), int(y1), int(x2), int(y2)), note), annotated


if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1])
    res, ann = analyze(img)
    print(res.to_dict())
    cv2.imwrite("plate_annotated.jpg", ann)
    print("saved plate_annotated.jpg")
