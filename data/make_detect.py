"""Детекционный датасет для YOLO: вклеиваем СВОИ синтетические UZ-номера в
РЕАЛЬНЫЕ фото авто (OLX) со случайным масштабом/позицией/перспективой/светом.
Bbox известен точно -> авторазметка. Данные полностью наши (композит).

Выход: images/{train,val}/*.jpg + labels/{train,val}/*.txt (YOLO) + data.yaml

Запуск:  python data/make_detect.py --per 4
"""
from __future__ import annotations
import argparse, glob, os, random
import cv2
import numpy as np


def warp_plate(plate):
    h, w = plate.shape[:2]
    m = random.uniform(0.0, 0.18)
    d = lambda: random.uniform(-m, m)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w*d(), h*d()], [w*(1+d()), h*d()],
                      [w*(1+d()), h*(1+d())], [w*d(), h*(1+d())]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(plate, M, (w, h), borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(np.ones((h, w), np.uint8)*255, M, (w, h))
    return warped, mask


def paste(bg, plate):
    H, W = bg.shape[:2]
    pw = int(W * random.uniform(0.10, 0.34))
    ph = int(pw * plate.shape[0] / plate.shape[1])
    if ph < 10 or ph > H*0.5:
        return None
    pl = cv2.resize(plate, (pw, ph))
    # лёгкая подгонка яркости под фон
    pl = np.clip(pl.astype(np.float32) * random.uniform(0.6, 1.05)
                 + random.uniform(-25, 15), 0, 255).astype(np.uint8)
    warped, mask = warp_plate(pl)
    x = random.randint(0, W - pw)
    y = random.randint(int(H*0.3), H - ph)      # плейт обычно в нижней части
    roi = bg[y:y+ph, x:x+pw]
    m3 = (mask[..., None] > 0)
    roi[:] = np.where(m3, warped, roi)
    if random.random() < 0.4:
        bg[y:y+ph, x:x+pw] = cv2.GaussianBlur(bg[y:y+ph, x:x+pw], (3, 3), 0)
    # bbox по непустой области маски
    ys, xs = np.where(mask > 0)
    bx1, by1, bx2, by2 = x+xs.min(), y+ys.min(), x+xs.max(), y+ys.max()
    cx = (bx1+bx2)/2/W; cy = (by1+by2)/2/H
    bw = (bx2-bx1)/W; bh = (by2-by1)/H
    return (cx, cy, bw, bh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=4)
    ap.add_argument("--bg", default="data/olx_raw")
    ap.add_argument("--plates", default="data/synth")
    ap.add_argument("--out", default="data/detect")
    args = ap.parse_args()
    bgs = sorted(glob.glob(os.path.join(args.bg, "*.jpg")))
    plates = sorted(glob.glob(os.path.join(args.plates, "*.png")))
    random.seed(1); random.shuffle(bgs)
    for split in ("train", "val"):
        os.makedirs(os.path.join(args.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", split), exist_ok=True)
    n_val = max(40, len(bgs)//10)
    val_bgs = set(bgs[:n_val])
    idx = 0
    for bg_path in bgs:
        bg0 = cv2.imread(bg_path)
        if bg0 is None:
            continue
        split = "val" if bg_path in val_bgs else "train"
        for _ in range(args.per):
            bg = bg0.copy()
            box = paste(bg, cv2.imread(random.choice(plates)))
            if box is None:
                continue
            fn = f"d{idx:05d}"
            cv2.imwrite(os.path.join(args.out, "images", split, fn+".jpg"), bg)
            with open(os.path.join(args.out, "labels", split, fn+".txt"), "w") as f:
                f.write(f"0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")
            idx += 1
    yaml = (f"path: {os.path.abspath(args.out)}\n"
            "train: images/train\nval: images/val\nnames:\n  0: plate\n")
    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        f.write(yaml)
    print(f"Композитов: {idx}  -> {args.out}  (data.yaml готов)")


if __name__ == "__main__":
    main()
