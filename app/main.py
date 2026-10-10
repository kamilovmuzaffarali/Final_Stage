"""PlateVision — FastAPI: фото авто -> номер (UZ) + талон приёма + журнал.

Запуск:  uvicorn app.main:app --host 0.0.0.0 --port 8860
"""
from __future__ import annotations
import os, sys, base64
import cv2, numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_ROOT)
from src import pipeline_plate as pipe  # noqa: E402
from src import journal  # noqa: E402

app = FastAPI(title="PlateVision", version="0.1")
_samples = os.path.join(_ROOT, "web", "samples")
if os.path.isdir(_samples):
    app.mount("/samples", StaticFiles(directory=_samples), name="samples")


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(_ROOT, "web", "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/log")
def log(limit: int = 20):
    return JSONResponse(journal.recent(limit))


@app.get("/stats")
def stats():
    return JSONResponse(journal.stats())


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "bad image"}, status_code=400)
    res, annotated = pipe.analyze(img)
    ok, buf = cv2.imencode(".jpg", annotated)
    out = res.to_dict()
    out["annotated_image"] = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    try:
        tk = journal.record(out["text"], out["decision"], out["ocr_conf"])
        out["ticket"] = tk["ticket"]; out["ts"] = tk["ts"]
    except Exception:
        pass
    return JSONResponse(out)
