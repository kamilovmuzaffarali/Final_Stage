"""Журнал осмотров на воротах — лёгкое операционное хранилище (SQLite).

Каждый осмотр = одна запись: талон (id), время, вердикт, крен, заметка.
Это то, что делает из «демки алгоритма» рабочий инструмент диспетчера:
история смены + счётчики GO/WARN/NO-GO.

Хранилище — один файл data/journal.db (gitignored). Без внешних зависимостей.
"""
from __future__ import annotations
import os
import sqlite3
import datetime as _dt
from contextlib import closing

_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "journal.db")


def _conn():
    os.makedirs(os.path.dirname(_DB), exist_ok=True)
    c = sqlite3.connect(_DB)
    c.row_factory = sqlite3.Row
    return c


def init():
    with closing(_conn()) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket    TEXT NOT NULL,
                ts        TEXT NOT NULL,
                decision  TEXT NOT NULL,
                tilt_deg  REAL,
                truck     TEXT,
                note      TEXT
            )
        """)
        c.commit()


def _ticket_id(dt: _dt.datetime, seq: int) -> str:
    # Человекочитаемый номер талона: LSG-ГГММДД-NNNN
    return f"LSG-{dt:%y%m%d}-{seq:04d}"


def record(decision: str, tilt_deg: float, note: str = "",
           truck: str = "") -> dict:
    """Пишет осмотр и возвращает талон (ticket, ts)."""
    init()
    now = _dt.datetime.now()
    with closing(_conn()) as c:
        cur = c.execute(
            "SELECT COUNT(*)+1 AS n FROM inspections WHERE substr(ts,1,10)=?",
            (now.strftime("%Y-%m-%d"),))
        seq = cur.fetchone()["n"]
        ticket = _ticket_id(now, seq)
        c.execute(
            "INSERT INTO inspections(ticket,ts,decision,tilt_deg,truck,note) "
            "VALUES(?,?,?,?,?,?)",
            (ticket, now.isoformat(timespec="seconds"), decision,
             round(float(tilt_deg), 2), truck, note))
        c.commit()
    return {"ticket": ticket, "ts": now.isoformat(timespec="seconds")}


def recent(limit: int = 20) -> list[dict]:
    init()
    with closing(_conn()) as c:
        rows = c.execute(
            "SELECT ticket,ts,decision,tilt_deg,note FROM inspections "
            "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def stats(day: str | None = None) -> dict:
    """Счётчики за день (по умолчанию — сегодня)."""
    init()
    day = day or _dt.date.today().isoformat()
    out = {"day": day, "GO": 0, "WARN": 0, "NO-GO": 0,
           "CANNOT-ASSESS": 0, "NO-DATA": 0, "total": 0}
    with closing(_conn()) as c:
        rows = c.execute(
            "SELECT decision, COUNT(*) n FROM inspections "
            "WHERE substr(ts,1,10)=? GROUP BY decision", (day,)).fetchall()
    for r in rows:
        out[r["decision"]] = r["n"]
        out["total"] += r["n"]
    out["blocked"] = out["NO-GO"] + out["WARN"]
    return out
