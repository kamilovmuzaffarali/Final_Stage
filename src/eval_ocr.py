"""Оценка OCR на удержанном СИНТЕТИЧЕСКОМ тест-сете (свой тест-сет).

Меряем seq-acc (номер целиком) и char-acc. Для честности тест берётся из того же
генератора, но с seed'ом отдельно и БЕЗ обучения на нём (train_ocr делит по shuffle).
Дополнительно — метрика на размеченных реальных кропах, если есть data/real_labels.csv.

Запуск:  python src/eval_ocr.py --data data/synth --n 1000
"""
from __future__ import annotations
import argparse, csv, os, random
import numpy as np, cv2, torch
from src.ocr_model import CRNN, ctc_greedy_decode

IMG_H, IMG_W = 32, 128
_dev = "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(path="weights/ocr_crnn.pt"):
    m = CRNN().to(_dev).eval()
    m.load_state_dict(torch.load(path, map_location=_dev)["model"])
    return m


def rec(m, img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    g = cv2.resize(g, (IMG_W, IMG_H)).astype(np.float32)
    g = (g / 255.0 - 0.5) / 0.5
    x = torch.from_numpy(g).unsqueeze(0).unsqueeze(0).to(_dev)
    with torch.no_grad():
        return ctc_greedy_decode(m(x).cpu())[0]


def score(preds, gts):
    seq = sum(p == g for p, g in zip(preds, gts)) / max(len(gts), 1)
    tot_ch = sum(len(g) for g in gts)
    ok_ch = sum(sum(a == b for a, b in zip(p, g)) for p, g in zip(preds, gts))
    return seq, ok_ch / max(tot_ch, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth")
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    m = load_model()

    rows = list(csv.DictReader(open(os.path.join(args.data, "labels.csv"))))
    random.seed(123); random.shuffle(rows); rows = rows[:args.n]
    preds, gts = [], []
    for r in rows:
        img = cv2.imread(os.path.join(args.data, r["file"]))
        preds.append(rec(m, img)); gts.append(r["text"])
    seq, ch = score(preds, gts)
    print(f"SYNTHETIC test (n={len(gts)}): seq-acc {seq:.3f}  char-acc {ch:.3f}")

    real = "data/real_labels.csv"
    if os.path.exists(real):
        rr = list(csv.DictReader(open(real)))
        rp, rg = [], []
        for r in rr:
            img = cv2.imread(r["file"])
            if img is None:
                continue
            rp.append(rec(m, img)); rg.append(r["text"].replace(" ", ""))
        s, c = score(rp, rg)
        print(f"REAL test (n={len(rg)}): seq-acc {s:.3f}  char-acc {c:.3f}")


if __name__ == "__main__":
    main()
