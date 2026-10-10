"""FastAPI-сервис: веб-аплоад фото грузовика -> решение GO/WARN/NO-GO + аннотация.

Запуск локально:
    pip install -r requirements.txt
    uvicorn app.main:app --host 0.0.0.0 --port 8000
Открыть http://localhost:8000
"""
from __future__ import annotations
import io
import os
import sys
import base64

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_ROOT)
from src import pipeline  # noqa: E402
from src import journal  # noqa: E402

app = FastAPI(title="Load Safety Gate", version="0.1")
app.mount("/samples", StaticFiles(directory=os.path.join(_ROOT, "web", "samples")),
          name="samples")


@app.get("/", response_class=HTMLResponse)
def index():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "index.html")
    with open(path, encoding="utf-8") as f:
        return f.read()


@app.get("/deck", response_class=HTMLResponse)
def deck():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "deck.html")
    with open(path, encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():
    return {"status": "ok"}


_EXAMPLES_CACHE = None


def _run_example(path):
    img = cv2.imread(path)
    if img is None:
        return None
    result, annotated = pipeline.analyze(img)
    ok, buf = cv2.imencode(".jpg", annotated)
    b64 = base64.b64encode(buf.tobytes()).decode()
    out = result.to_dict()
    out["annotated_image"] = f"data:image/jpeg;base64,{b64}"
    return out


@app.get("/examples")
def examples():
    """Готовые примеры-выводы (норма/перекос) — считаются один раз реальным пайплайном."""
    global _EXAMPLES_CACHE
    if _EXAMPLES_CACHE is None:
        sdir = os.path.join(_ROOT, "web", "samples")
        _EXAMPLES_CACHE = []
        for name, title in (("normal.jpg", "Норма"),
                            ("tilted.jpg", "Перекос (синт. тест 10°)")):
            r = _run_example(os.path.join(sdir, name))
            if r:
                r["title"] = title
                _EXAMPLES_CACHE.append(r)
    return JSONResponse(_EXAMPLES_CACHE)


@app.get("/log")
def log(limit: int = 20):
    """Журнал последних осмотров (операционная история смены)."""
    return JSONResponse(journal.recent(limit))


@app.get("/stats")
def stats():
    """Счётчики за сегодня: всего/GO/WARN/NO-GO/не оценить."""
    return JSONResponse(journal.stats())


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "bad image"}, status_code=400)
    result, annotated = pipeline.analyze(img)
    ok, buf = cv2.imencode(".jpg", annotated)
    b64 = base64.b64encode(buf.tobytes()).decode()
    out = result.to_dict()
    out["annotated_image"] = f"data:image/jpeg;base64,{b64}"
    # записываем осмотр в журнал -> получаем номер талона (операционный слой)
    try:
        tk = journal.record(out["decision"], out.get("tilt_deg", 0.0),
                            out.get("note", ""))
        out["ticket"] = tk["ticket"]
        out["ts"] = tk["ts"]
    except Exception:
        pass
    return JSONResponse(out)
