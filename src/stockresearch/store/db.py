"""SQLite persistence for daily runs. Stores full reports as JSON plus flat rows for
trend/track-record queries. Stdlib sqlite3 only — no ORM needed for this shape."""

from __future__ import annotations

import json
import sqlite3
from datetime import date as Date
from pathlib import Path

from ..config import load_settings
from ..models import DailyReport, Recommendation

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_date    TEXT PRIMARY KEY,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP,
    universe_size INTEGER,
    macro_summary TEXT,
    report_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ticker_analysis (
    run_date    TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    name        TEXT,
    sector      TEXT,
    lean        TEXT,
    confidence  REAL,
    score       REAL,
    last_price  REAL,
    beta        REAL,
    alpha       REAL,
    sharpe      REAL,
    PRIMARY KEY (run_date, ticker)
);
CREATE TABLE IF NOT EXISTS macro_signals (
    run_date    TEXT NOT NULL,
    headline    TEXT,
    impact      TEXT,
    confidence  REAL,
    sectors     TEXT,
    affected_tickers TEXT,
    sources     TEXT
);
CREATE TABLE IF NOT EXISTS outcomes (
    run_date    TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    lean        TEXT,
    confidence  REAL,
    price_then  REAL,
    price_later REAL,
    forward_return REAL,
    correct     INTEGER,
    PRIMARY KEY (run_date, ticker, horizon_days)
);
CREATE TABLE IF NOT EXISTS recommendations (
    run_date    TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    action      TEXT,
    shares      REAL,
    amount      REAL,
    target_pct  REAL,
    price_at_rec REAL,
    confidence  REAL,
    lean        TEXT,
    score       REAL,
    rationale   TEXT,
    notes       TEXT,
    PRIMARY KEY (run_date, ticker)
);
CREATE TABLE IF NOT EXISTS rec_outcomes (
    run_date    TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    action      TEXT,
    confidence  REAL,
    price_then  REAL,
    price_later REAL,
    forward_return REAL,
    win         INTEGER,
    PRIMARY KEY (run_date, ticker, horizon_days)
);
CREATE INDEX IF NOT EXISTS idx_ta_ticker ON ticker_analysis(ticker);
CREATE INDEX IF NOT EXISTS idx_rec_ticker ON recommendations(ticker);
"""


def _connect() -> sqlite3.Connection:
    path: Path = load_settings().db_full_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Idempotent column adds for DBs created before a column existed."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(macro_signals)")}
    if "sources" not in cols:
        conn.execute("ALTER TABLE macro_signals ADD COLUMN sources TEXT")


def save_report(report: DailyReport) -> None:
    rd = report.run_date.isoformat()
    with _connect() as conn:
        conn.execute("DELETE FROM runs WHERE run_date = ?", (rd,))
        conn.execute("DELETE FROM ticker_analysis WHERE run_date = ?", (rd,))
        conn.execute("DELETE FROM macro_signals WHERE run_date = ?", (rd,))
        conn.execute(
            "INSERT INTO runs (run_date, universe_size, macro_summary, report_json) "
            "VALUES (?, ?, ?, ?)",
            (rd, report.universe_size, report.macro_summary, report.model_dump_json()),
        )
        for a in report.analyses:
            conn.execute(
                "INSERT INTO ticker_analysis (run_date, ticker, name, sector, lean, "
                "confidence, score, last_price, beta, alpha, sharpe) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    rd, a.ticker, a.name, a.sector, a.lean.value, a.confidence, a.score,
                    a.metrics.last_price, a.metrics.beta, a.metrics.alpha, a.metrics.sharpe,
                ),
            )
        for sgl in report.macro_signals:
            conn.execute(
                "INSERT INTO macro_signals (run_date, headline, impact, confidence, "
                "sectors, affected_tickers, sources) VALUES (?,?,?,?,?,?,?)",
                (
                    rd, sgl.headline, sgl.impact.value, sgl.confidence,
                    json.dumps(sgl.sectors), json.dumps(sgl.affected_tickers),
                    json.dumps([s.url for s in sgl.sources if s.url]),
                ),
            )


def load_report(run_date: Date) -> DailyReport | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT report_json FROM runs WHERE run_date = ?", (run_date.isoformat(),)
        ).fetchone()
    return DailyReport.model_validate_json(row["report_json"]) if row else None


def latest_report() -> DailyReport | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT report_json FROM runs ORDER BY run_date DESC LIMIT 1"
        ).fetchone()
    return DailyReport.model_validate_json(row["report_json"]) if row else None


def list_runs() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT run_date, universe_size, created_at FROM runs ORDER BY run_date DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def ticker_history(ticker: str) -> list[dict]:
    """Per-run history for one ticker — the basis of the track-record view."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT run_date, lean, confidence, score, last_price, beta, alpha, sharpe "
            "FROM ticker_analysis WHERE ticker = ? ORDER BY run_date",
            (ticker.upper(),),
        ).fetchall()
    return [dict(r) for r in rows]


def all_calls() -> list[dict]:
    """Every stored per-ticker call (the basis for outcome scoring)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT run_date, ticker, lean, confidence, last_price FROM ticker_analysis "
            "ORDER BY run_date"
        ).fetchall()
    return [dict(r) for r in rows]


def upsert_outcome(
    run_date: str, ticker: str, horizon_days: int, lean: str, confidence: float | None,
    price_then: float | None, price_later: float | None, forward_return: float | None,
    correct: bool | None,
) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO outcomes (run_date, ticker, horizon_days, lean, "
            "confidence, price_then, price_later, forward_return, correct) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (run_date, ticker, horizon_days, lean, confidence, price_then, price_later,
             forward_return, None if correct is None else int(correct)),
        )


def outcomes_rows() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM outcomes WHERE forward_return IS NOT NULL").fetchall()
    return [dict(r) for r in rows]


def save_recommendation(rec: Recommendation) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO recommendations (run_date, ticker, action, shares, amount, "
            "target_pct, price_at_rec, confidence, lean, score, rationale, notes) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                rec.run_date.isoformat(), rec.ticker, rec.action.value, rec.shares, rec.amount,
                rec.target_pct, rec.price_at_rec, rec.confidence, rec.lean.value, rec.score,
                rec.rationale, json.dumps(rec.notes),
            ),
        )


def recommendations_rows(ticker: str | None = None) -> list[dict]:
    sql = "SELECT * FROM recommendations"
    params: tuple = ()
    if ticker:
        sql += " WHERE ticker = ?"
        params = (ticker.upper(),)
    sql += " ORDER BY run_date DESC, ticker"
    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def save_rec_outcome(
    run_date: str, ticker: str, horizon_days: int, action: str, confidence: float | None,
    price_then: float | None, price_later: float | None, forward_return: float | None,
    win: bool | None,
) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO rec_outcomes (run_date, ticker, horizon_days, action, "
            "confidence, price_then, price_later, forward_return, win) VALUES (?,?,?,?,?,?,?,?,?)",
            (run_date, ticker, horizon_days, action, confidence, price_then, price_later,
             forward_return, None if win is None else int(win)),
        )


def rec_outcomes_rows() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM rec_outcomes WHERE forward_return IS NOT NULL"
        ).fetchall()
    return [dict(r) for r in rows]


def top_picks(run_date: Date, limit: int = 10) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT ticker, name, lean, confidence, score FROM ticker_analysis "
            "WHERE run_date = ? ORDER BY score DESC LIMIT ?",
            (run_date.isoformat(), limit),
        ).fetchall()
    return [dict(r) for r in rows]
