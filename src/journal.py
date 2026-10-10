"""Журнал приёма авто по номеру (SQLite) — операционный слой продукта.

Каждое сканирование = запись: талон, время, номер, вердикт, уверенность.
Даёт историю смены + счётчики (прочитано / не прочитано) для дилера/сервиса.
"""
from __future__ import annotations
import os, sqlite3, datetime as _dt
from contextlib import closing

_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "journal.db")


def _conn():
    os.makedirs(os.path.dirname(_DB), exist_ok=True)
    c = sqlite3.connect(_DB); c.row_factory = sqlite3.Row
    return c


def init():
    with closing(_conn()) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS scans(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket TEXT, ts TEXT, plate TEXT, decision TEXT, conf REAL)""")
        c.commit()


def record(plate: str, decision: str, conf: float) -> dict:
    init()
    now = _dt.datetime.now()
    with closing(_conn()) as c:
        seq = c.execute("SELECT COUNT(*)+1 n FROM scans WHERE substr(ts,1,10)=?",
                        (now.strftime("%Y-%m-%d"),)).fetchone()["n"]
        ticket = f"PV-{now:%y%m%d}-{seq:04d}"
        c.execute("INSERT INTO scans(ticket,ts,plate,decision,conf) VALUES(?,?,?,?,?)",
                  (ticket, now.isoformat(timespec="seconds"), plate, decision,
                   round(float(conf), 3)))
        c.commit()
    return {"ticket": ticket, "ts": now.isoformat(timespec="seconds")}


def recent(limit: int = 20) -> list[dict]:
    init()
    with closing(_conn()) as c:
        rows = c.execute("SELECT ticket,ts,plate,decision,conf FROM scans "
                         "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def stats(day: str | None = None) -> dict:
    init()
    day = day or _dt.date.today().isoformat()
    out = {"day": day, "READ": 0, "CANNOT-READ": 0, "NO-PLATE": 0, "total": 0}
    with closing(_conn()) as c:
        for r in c.execute("SELECT decision,COUNT(*) n FROM scans "
                           "WHERE substr(ts,1,10)=? GROUP BY decision", (day,)):
            out[r["decision"]] = r["n"]; out["total"] += r["n"]
    return out
