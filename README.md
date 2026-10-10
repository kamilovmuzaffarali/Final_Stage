# PlateVision — UZ License Plate Recognition System

**Live demo:** https://platevision.astrotms.com

Automatic license plate recognition for Uzbek vehicles (UZ format).  
Our own trained models + our own dataset — no external OCR APIs called.

---

## What it does

1. **Detect** the plate in a car photo — our YOLOv8n detector trained on our composites.
2. **Recognise** the text — our CRNN+CTC model trained on our UZ synthetic plates.
3. **Validate** the UZ format (`RR L DDD LL` or `RR DDD LLL`).
4. **Gate** uncertain results → says "CANNOT READ" instead of guessing.
5. **Issue a ticket** PV-YYMMDD-NNNN with timestamp + shift journal for the operator.

---

## Demo use

Upload a photo of a car or a cropped plate image at https://platevision.astrotms.com  
The system returns a ticket with the plate number and a shift dashboard.

---

## Architecture

```
Photo → YOLOv8n plate detector → crop
                                  ↓
                        CRNN + CTC OCR (our model)
                                  ↓
                  UZ format regex + confidence gate
                                  ↓
                       READ / CANNOT-READ / NO-PLATE
                                  ↓
                     Ticket (SQLite journal)
```

Models trained entirely on **our own dataset** (see [Data Sources](#data-sources--licences)).

---

## Run locally

```bash
git clone https://github.com/kamilovmuzaffarali/Final_Stage.git
git checkout platevision
pip install -r requirements.txt

# Reproduce dataset (optional — weights already in weights/)
python data/gen_synth.py --n 7000 --out data/synth
python data/harvest_olx.py --n 500 --out data/olx_raw
python data/make_detect.py --per 4

# Train models (optional — weights already trained)
python src/train_ocr.py --data data/synth --epochs 22
# YOLOv8n detector: see data/make_detect.py + ultralytics train

# Start API
PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 8860
# Open http://localhost:8860
```

**System requirements:** macOS/Linux, Python 3.9+, ~4 GB RAM.  
GPU: MPS (Apple Silicon) or CUDA; CPU fallback supported.

---

## Performance

| Split | Seq-Acc | Char-Acc |
|---|---|---|
| Synthetic val (n=900) | **0.999** | **1.000** |
| Real plates (clean crop) | ~0.60–0.75 | ~0.90–0.95 |

*Real-plate gap is expected without real labeled training data.  
The "CANNOT READ" gate prevents wrong outputs in uncertain cases.*

---

## Data Sources & Licences

See full details in [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

| # | Source | Type | Licence |
|---|---|---|---|
| 1 | Own synthetic UZ plates (`data/gen_synth.py`) | OCR training | Own — created for this project |
| 2 | OLX.uz public car listings (`data/harvest_olx.py`) | Real car backgrounds | Public listings, no personal data; manifest at `data/olx_raw/sources.csv` |
| 3 | Composite detection dataset (`data/make_detect.py`) | Detection training | Derived from 1+2 above |
| 4 | YOLOv8n backbone weights (COCO) | Pre-trained base | AGPL-3.0 (Ultralytics) |

No faces or personal data are collected. Plates in OLX listings are part of publicly posted adverts.

---

## Repository layout

```
├── data/
│   ├── gen_synth.py       # synthetic UZ plate generator (9 fonts, PIL)
│   ├── harvest_olx.py     # OLX car photo harvester (public listings)
│   └── make_detect.py     # composite detection dataset builder
├── src/
│   ├── ocr_model.py       # CRNN + CTC architecture
│   ├── train_ocr.py       # OCR training script
│   ├── pipeline_plate.py  # end-to-end pipeline
│   ├── journal.py         # SQLite intake log
│   └── eval_ocr.py        # evaluation script
├── app/
│   └── main.py            # FastAPI web service
├── web/
│   └── index.html         # operator dashboard UI
├── weights/               # trained model weights (gitignored — download below)
├── docs/
│   └── DATA_SOURCES.md    # full data sources and licences
├── QUESTIONS.md           # open decisions log
└── requirements.txt
```

> **Note:** `weights/` and `data/olx_raw/`, `data/synth/`, `data/detect/` are gitignored  
> (binary blobs and large datasets). Reproduce with the scripts above.

---

## Team

Kamilov Muzaffarali — Computer Vision track, Toyota / BYD hackathon 2026
