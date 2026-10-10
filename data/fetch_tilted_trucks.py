"""Качает фото ПЕРЕКОШЕННЫХ/наклонённых грузовиков из открытых источников
(Wikimedia Commons) — для NO-GO кейсов и теста точности крена.

Сохраняет в data/images/tilted/ , манифест data/tilted_sources.csv (с лицензией).
Запуск:  python data/fetch_tilted_trucks.py --n 40
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
QUERIES = ["overturned truck", "truck rollover", "tipped over truck",
           "truck on slope", "leaning truck", "truck accident tilted",
           "lorry overturned", "truck crash on side", "truck stuck tilted",
           "dump truck tipping"]

OUT_DIR = os.path.join(os.path.dirname(__file__), "images", "tilted")
MANIFEST = os.path.join(os.path.dirname(__file__), "tilted_sources.csv")


def _get(params):
    import json
    req = Request(API + "?" + urlencode(params), headers=UA)
    return json.load(urlopen(req, timeout=30))


def search_images(query, limit):
    params = {"action": "query", "format": "json", "generator": "search",
              "gsrsearch": f"filetype:bitmap {query}", "gsrlimit": limit,
              "gsrnamespace": 6, "prop": "imageinfo",
              "iiprop": "url|extmetadata", "iiurlwidth": 1024}
    data = _get(params)
    pages = (data.get("query") or {}).get("pages", {})
    out = []
    for p in pages.values():
        ii = (p.get("imageinfo") or [{}])[0]
        url = ii.get("thumburl") or ii.get("url")
        lic = ((ii.get("extmetadata", {}) or {}).get("LicenseShortName", {}) or {}).get("value", "unknown")
        if url and url.split("?")[0].lower().endswith((".jpg", ".jpeg", ".png")):
            out.append((p.get("title", ""), url, lic))
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
            for title, url, lic in search_images(q, 12):
                if url in seen or len(rows) >= args.n:
                    continue
                seen.add(url)
                fname = f"tilt_{len(rows):03d}.jpg"
                try:
                    img = urlopen(Request(url, headers=UA), timeout=30).read()
                    with open(os.path.join(OUT_DIR, fname), "wb") as f:
                        f.write(img)
                    rows.append((fname, title, url, lic))
                    print(f"[{len(rows)}] {fname}  {lic}")
                except Exception as e:
                    print("skip", str(e)[:50])
                time.sleep(0.3)
        except Exception as e:
            print("query failed", q, str(e)[:50])
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "commons_title", "source_url", "license"])
        w.writerows(rows)
    print(f"\nСкачано {len(rows)} -> {OUT_DIR}\nМанифест: {MANIFEST}")


if __name__ == "__main__":
    main()
