"""Обучение CRNN на нашем синтетическом датасете UZ-номеров (CTC).

Аугментации на лету (перспектива/блюр/шум/свет/поворот) закрывают разрыв
синтетика→реальные фото. Делим на train/val, меряем char-acc и seq-acc (точный
номер целиком). Сохраняем лучший чекпойнт weights/ocr_crnn.pt.

Запуск:  python src/train_ocr.py --data data/synth --epochs 12
"""
from __future__ import annotations
import argparse, csv, os, random, math
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ocr_model import CRNN, encode, ctc_greedy_decode, NUM_CLASSES

IMG_H, IMG_W = 32, 128


def aug(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    # перспектива
    if random.random() < 0.7:
        m = 0.12
        d = lambda: random.uniform(-m, m)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[w*d(), h*d()], [w*(1+d()), h*d()],
                          [w*(1+d()), h*(1+d())], [w*d(), h*(1+d())]])
        M = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M, (w, h), borderValue=(127, 127, 127))
    if random.random() < 0.3:                      # поворот
        a = random.uniform(-6, 6)
        M = cv2.getRotationMatrix2D((w/2, h/2), a, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=(127, 127, 127))
    if random.random() < 0.5:                      # блюр
        k = random.choice([3, 3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if random.random() < 0.6:                      # яркость/контраст
        al = random.uniform(0.5, 1.4); be = random.uniform(-40, 40)
        img = np.clip(img.astype(np.float32)*al + be, 0, 255).astype(np.uint8)
    if random.random() < 0.5:                      # шум
        n = np.random.normal(0, random.uniform(4, 18), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32)+n, 0, 255).astype(np.uint8)
    if random.random() < 0.4:                      # JPEG-деградация
        q = random.randint(25, 70)
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE if img.ndim == 2 else cv2.IMREAD_COLOR)
    return img


class Plates(Dataset):
    def __init__(self, rows, root, train=True):
        self.rows, self.root, self.train = rows, root, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        fn, text = self.rows[i]
        img = cv2.imread(os.path.join(self.root, fn), cv2.IMREAD_GRAYSCALE)
        if self.train:
            img = aug(img)
        img = cv2.resize(img, (IMG_W, IMG_H))
        img = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
        return torch.from_numpy(img).unsqueeze(0), text


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    targets, lengths = [], []
    for _, t in batch:
        e = encode(t); targets += e; lengths.append(len(e))
    return imgs, torch.tensor(targets), torch.tensor(lengths), [b[1] for b in batch]


def evaluate(model, loader, device):
    model.eval()
    ok_seq = tot = ok_ch = tot_ch = 0
    with torch.no_grad():
        for imgs, _, _, texts in loader:
            logits = model(imgs.to(device))
            preds = ctc_greedy_decode(logits.cpu())
            for p, t in zip(preds, texts):
                tot += 1; ok_seq += (p == t)
                tot_ch += len(t)
                ok_ch += sum(1 for a, b in zip(p, t) if a == b)
    return ok_seq / max(tot, 1), ok_ch / max(tot_ch, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=128)
    args = ap.parse_args()

    with open(os.path.join(args.data, "labels.csv")) as f:
        rows = [(r["file"], r["text"]) for r in csv.DictReader(f)]
    random.seed(0); random.shuffle(rows)
    n_val = max(200, len(rows) // 10)
    val, train = rows[:n_val], rows[n_val:]
    print(f"train {len(train)}  val {len(val)}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tr = DataLoader(Plates(train, args.data, True), args.bs, shuffle=True,
                    collate_fn=collate, num_workers=0)
    va = DataLoader(Plates(val, args.data, False), args.bs, shuffle=False,
                    collate_fn=collate, num_workers=0)

    model = CRNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=6, gamma=0.3)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    os.makedirs("weights", exist_ok=True)
    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train(); tl = 0; nb = 0
        for imgs, targets, lengths, _ in tr:
            imgs = imgs.to(device)
            logits = model(imgs)                      # (T,B,C)
            T = logits.size(0)
            logp = logits.log_softmax(2).cpu()        # CTC на CPU (MPS не умеет)
            in_len = torch.full((imgs.size(0),), T, dtype=torch.long)
            loss = ctc(logp, targets, in_len, lengths)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tl += loss.item(); nb += 1
        sched.step()
        seq, ch = evaluate(model, va, device)
        print(f"ep {ep:2d}  loss {tl/max(nb,1):.3f}  val seq-acc {seq:.3f}  char-acc {ch:.3f}")
        if seq >= best:
            best = seq
            torch.save({"model": model.state_dict(), "chars": True}, "weights/ocr_crnn.pt")
    print(f"BEST val seq-acc {best:.3f}  -> weights/ocr_crnn.pt")


if __name__ == "__main__":
    main()
