"""Генератор СИНТЕТИЧЕСКИХ узбекских автономеров (наш датасет — наш ров).

Рисуем номер по формату UZ на белой пластине с левой синей полосой "UZ",
затем сохраняем чистый кроп + подпись (ground-truth текст). Аугментации
(перспектива, шум, блюр, свет) добавляются на этапе обучения — так один
синтетик даёт много вариаций и метрика честная.

Формат UZ (частные авто), регион 01–90:
  тип A:  RR L DDD LL     напр. 01 A 123 BC
  тип B:  RR DDD LLL      напр. 01 777 ABC
Буквы — латиница без похожих на цифры (без I,O,Q).

Запуск:  python data/gen_synth.py --n 4000 --out data/synth
"""
from __future__ import annotations
import argparse, csv, os, random, string
from PIL import Image, ImageDraw, ImageFont

import os as _os
LETTERS = "ABCDEFGHJKLMNPRSTUVXYZ"   # без I,O,Q (путаница с 1,0)
DIGITS = "0123456789"
FONT_UZ = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
_SUP = "/System/Library/Fonts/Supplemental/"
# Много разных шрифтов -> OCR учит форму букв инвариантно к шрифту и
# обобщается на реальный шрифт номеров (DIN — почти «номерной»).
FONTS = [p for p in [
    _SUP+"DIN Condensed Bold.ttf", _SUP+"DIN Alternate Bold.ttf",
    _SUP+"Arial Bold.ttf", _SUP+"Arial Black.ttf", _SUP+"Arial Narrow Bold.ttf",
    _SUP+"Tahoma Bold.ttf", _SUP+"Verdana Bold.ttf", _SUP+"Trebuchet MS Bold.ttf",
    _SUP+"Impact.ttf",
] if _os.path.exists(p)]
_FW = [3, 3, 1, 1, 1, 1, 1, 1, 1][:len(FONTS)]


def _pick_font():
    return random.choices(FONTS, weights=_FW, k=1)[0]


def plate_text() -> str:
    rr = f"{random.randint(1, 90):02d}"
    if random.random() < 0.5:
        return f"{rr} {random.choice(LETTERS)} {''.join(random.choices(DIGITS,k=3))} {''.join(random.choices(LETTERS,k=2))}"
    return f"{rr} {''.join(random.choices(DIGITS,k=3))} {''.join(random.choices(LETTERS,k=3))}"


def _flag_strip(d, x0, W, H):
    """Правая полоса: флаг UZ (голубой/белый/зелёный с красными линиями) + 'UZ'."""
    bw = W - x0
    # три горизонтальные полосы флага в верхней части
    fy = int(H * 0.18); fh = int(H * 0.44)
    band_h = fh // 3
    cols = [(0, 153, 219), (255, 255, 255), (30, 160, 70)]  # голубой/белый/зелёный
    for i, c in enumerate(cols):
        d.rectangle([x0 + 4, fy + i*band_h, W - 4, fy + (i+1)*band_h], fill=c)
    # тонкие красные разделители
    for i in (1, 2):
        d.line([x0 + 4, fy + i*band_h, W - 4, fy + i*band_h], fill=(200, 40, 40), width=1)
    fuz = ImageFont.truetype(FONT_UZ, int(H * 0.26))
    tb = d.textbbox((0, 0), "UZ", font=fuz)
    d.text((x0 + (bw - (tb[2]-tb[0]))//2 - 2, int(H*0.64)), "UZ", font=fuz, fill=(0, 0, 0))


def render(text: str, W=440, H=100) -> Image.Image:
    # лёгкая вариация тона пластины (не всегда чисто белая)
    base = random.choice([(255, 255, 255), (250, 250, 248), (245, 246, 243)])
    img = Image.new("RGB", (W, H), base)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W - 1, H - 1], outline=(10, 10, 10), width=3)
    # правый блок флага/UZ (~14% ширины)
    strip_x = int(W * 0.86)
    d.line([strip_x, 6, strip_x, H - 6], fill=(60, 60, 60), width=1)
    _flag_strip(d, strip_x, W, H)
    # основной номер: регион слева чуть меньше, затем группа крупно
    parts = text.split(" ")
    region = parts[0]
    rest = " ".join(parts[1:])
    area_w = strip_x
    fp = _pick_font()
    f_big = ImageFont.truetype(fp, int(H * random.uniform(0.56, 0.66)))
    f_reg = ImageFont.truetype(fp, int(H * 0.52))
    pad = int(W * 0.03)
    # регион
    rb = d.textbbox((0, 0), region, font=f_reg)
    ry = (H - (rb[3]-rb[1]))//2 - rb[1]
    d.text((pad, ry), region, font=f_reg, fill=(10, 10, 10))
    x_after = pad + (rb[2]-rb[0]) + int(W*0.02)
    # точка-разделитель как на реальных
    d.ellipse([x_after, H//2-3, x_after+6, H//2+3], fill=(10, 10, 10))
    x_after += int(W*0.02)
    # основная группа (с разрядкой), центрируем в оставшемся месте
    core = rest.replace(" ", " ")
    cb = d.textbbox((0, 0), core, font=f_big)
    cw, ch = cb[2]-cb[0], cb[3]-cb[1]
    cx = x_after + max(0, ((area_w - x_after) - cw)//2) - cb[0]
    cy = (H - ch)//2 - cb[1]
    d.text((cx, cy), core, font=f_big, fill=(10, 10, 10))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--out", default="data/synth")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for i in range(args.n):
        t = plate_text()
        img = render(t)
        fn = f"p{i:05d}.png"
        img.save(os.path.join(args.out, fn))
        rows.append((fn, t.replace(" ", "")))
    with open(os.path.join(args.out, "labels.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "text"]); w.writerows(rows)
    print(f"Сгенерировано {len(rows)} номеров -> {args.out}")
    print("Примеры:", [r[1] for r in rows[:6]])


if __name__ == "__main__":
    main()
