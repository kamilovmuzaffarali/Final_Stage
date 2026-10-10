"""Скачивает публичные фото грузовиков с ЯВНОЙ лицензией (Wikimedia Commons).

Каждый файл Commons несёт лицензию (CC-BY / CC-BY-SA / CC0 / Public domain),
которую мы фиксируем в data/public_sources.csv — для раздела «источники/лицензии».

Запуск:
    python data/fetch_public_trucks.py --n 40
Картинки -> data/images/public/ , манифест -> data/public_sources.csv
Изображения в git НЕ коммитим (.gitignore), манифест коммитим для воспроизводимости.
"""
from __future__ import annotations
import argparse
import csv
import os
import time
from urllib.parse import urlencode
from urllib.request import urlopen, Request

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "LoadSafetyGate-hackathon/1.0 (educational)"}
QUERIES = ["semi truck", "lorry truck", "cargo truck", "flatbed truck",
           "truck trailer", "dump truck"]

OUT_DIR = os.path.join(os.path.dirname(__file__), "images", "public")
MANIFEST = os.path.join(os.path.dirname(__file__), "public_sources.csv")


def _get(params):
    url = API + "?" + urlencode(params)
    req = Request(url, headers=UA)
    import json
    return json.load(urlopen(req, timeout=30))


def search_images(query, limit):
    params = {
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}", "gsrlimit": limit,
        "gsrnamespace": 6, "prop": "imageinfo",
        "iiprop": "url|extmetadata", "iiurlwidth": 1024,
    }
    data = _get(params)
    pages = (data.get("query") or {}).get("pages", {})
    out = []
    for p in pages.values():
        ii = (p.get("imageinfo") or [{}])[0]
        url = ii.get("thumburl") or ii.get("url")
        meta = ii.get("extmetadata", {})
        lic = (meta.get("LicenseShortName", {}) or {}).get("value", "unknown")
        desc = p.get("title", "")
        base = (url or "").split("?")[0].lower()
        if base.endswith((".jpg", ".jpeg", ".png")):
            out.append((desc, url, lic))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    seen, rows = set(), []
    for q in QUERIES:
        if len(rows) >= args.n:
            break
        try:
            for title, url, lic in search_images(q, 15):
                if url in seen or len(rows) >= args.n:
                    continue
                seen.add(url)
                fname = f"truck_{len(rows):03d}.jpg"
                try:
                    req = Request(url, headers=UA)
                    img = urlopen(req, timeout=30).read()
                    with open(os.path.join(OUT_DIR, fname), "wb") as f:
                        f.write(img)
                    rows.append((fname, title, url, lic))
                    print(f"[{len(rows)}] {fname}  {lic}")
                except Exception as e:
                    print("skip", url, str(e)[:60])
                time.sleep(0.3)
        except Exception as e:
            print("query failed", q, str(e)[:60])

    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "commons_title", "source_url", "license"])
        w.writerows(rows)
    print(f"\nСкачано {len(rows)} фото -> {OUT_DIR}\nМанифест: {MANIFEST}")


if __name__ == "__main__":
    main()
