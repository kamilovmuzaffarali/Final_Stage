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

LETTERS = "ABCDEFGHJKLMNPRSTUVXYZ"   # без I,O,Q (путаница с 1,0)
DIGITS = "0123456789"
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_UZ = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def plate_text() -> str:
    rr = f"{random.randint(1, 90):02d}"
    if random.random() < 0.5:
        return f"{rr} {random.choice(LETTERS)} {''.join(random.choices(DIGITS,k=3))} {''.join(random.choices(LETTERS,k=2))}"
    return f"{rr} {''.join(random.choices(DIGITS,k=3))} {''.join(random.choices(LETTERS,k=3))}"


def render(text: str, W=440, H=100) -> Image.Image:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W - 1, H - 1], outline="black", width=3)
    # левая синяя полоса "UZ"
    band = 52
    d.rectangle([3, 3, band, H - 4], fill=(0, 51, 160))
    fuz = ImageFont.truetype(FONT_UZ, 26)
    d.text((10, 14), "UZ", font=fuz, fill="white")
    fflag = ImageFont.truetype(FONT_UZ, 16)
    d.text((14, 54), "●", font=fflag, fill=(0, 180, 0))
    # основной текст (без пробелов-разделителей рисуем с разрядкой)
    f = ImageFont.truetype(FONT, 58)
    core = text.replace(" ", "  ")
    bb = d.textbbox((0, 0), core, font=f)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    x = band + ( (W - band) - tw )//2 - bb[0]
    y = (H - th)//2 - bb[1]
    d.text((x, y), core, font=f, fill="black")
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
