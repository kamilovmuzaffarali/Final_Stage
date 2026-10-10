"""Сбор РЕАЛЬНЫХ фото авто с OLX.uz (публичные объявления купли-продажи).

Источник задекларирован: публичные объявления OLX.uz, раздел «Легковые авто»
(category_id=108). Берём по несколько фото с объявления, пишем манифест с
исходным URL каждой картинки (для раздела «источники данных» по правилам).

Запуск:  python data/harvest_olx.py --n 400 --out data/olx_raw
"""
from __future__ import annotations
import argparse, csv, os, time
from curl_cffi import requests

API = "https://www.olx.uz/api/v1/offers/"
H = {"accept": "application/json", "accept-language": "ru"}
CAT_CARS = 108


def photo_url(link: str, w=1000, h=750) -> str:
    return link.replace("{width}", str(w)).replace("{height}", str(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default="data/olx_raw")
    ap.add_argument("--per", type=int, default=2, help="фото с одного объявления")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows, seen = [], set()
    offset = 0
    while len(rows) < args.n:
        url = f"{API}?offset={offset}&limit=40&category_id={CAT_CARS}"
        try:
            data = requests.get(url, impersonate="chrome", headers=H, timeout=30).json()
        except Exception as e:
            print("api err", str(e)[:60]); break
        offers = data.get("data", [])
        if not offers:
            break
        for o in offers:
            if len(rows) >= args.n:
                break
            for ph in (o.get("photos") or [])[:args.per]:
                if len(rows) >= args.n:
                    break
                u = photo_url(ph.get("link", ""))
                if not u or u in seen:
                    continue
                seen.add(u)
                fn = f"car_{len(rows):05d}.jpg"
                try:
                    img = requests.get(u, impersonate="chrome", timeout=30).content
                    with open(os.path.join(args.out, fn), "wb") as f:
                        f.write(img)
                    rows.append((fn, u, o.get("id"), o.get("url", "")))
                except Exception as e:
                    print("skip", str(e)[:40])
                time.sleep(0.15)
        offset += 40
        print(f"  собрано {len(rows)}...")
        time.sleep(0.3)
    with open(os.path.join(args.out, "sources.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "image_url", "offer_id", "offer_url"])
        w.writerows(rows)
    print(f"Готово: {len(rows)} фото -> {args.out} (манифест sources.csv)")


if __name__ == "__main__":
    main()
