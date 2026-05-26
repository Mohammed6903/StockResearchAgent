"""MongoDB backend, API-compatible with the SQLite `db` module.

Stores each DailyReport as one document (its natural shape — nested snapshots/analyses/
macro signals) in the `reports` collection, and scored outcomes in `outcomes`. The
analysis-derived queries (ticker history, calls, top picks) iterate the report docs in
Python — volumes are tiny (one doc per day), so this stays simple and correct.

Connection comes from the MONGODB_URI env var (e.g. an Atlas SRV string); never committed.
A `client` may be injected for testing.
"""

from __future__ import annotations

from datetime import date as Date
from datetime import datetime, timezone

from ..models import DailyReport


class MongoStore:
    def __init__(self, uri: str | None = None, dbname: str = "stockresearch", client=None):
        if client is None:
            from pymongo import MongoClient

            client = MongoClient(uri, serverSelectionTimeoutMS=4000, appname="stockresearch")
            client.admin.command("ping")  # fail fast so callers can fall back to SQLite
        self.client = client
        self.reports = client[dbname]["reports"]
        self.outcomes = client[dbname]["outcomes"]
        self.reports.create_index("run_date", unique=True)
        self.outcomes.create_index(
            [("run_date", 1), ("ticker", 1), ("horizon_days", 1)], unique=True
        )

    # ---- reports -----------------------------------------------------------------------
    def save_report(self, report: DailyReport) -> None:
        doc = report.model_dump(mode="json")
        doc["_id"] = report.run_date.isoformat()
        doc["created_at"] = datetime.now(timezone.utc).isoformat()
        self.reports.replace_one({"_id": doc["_id"]}, doc, upsert=True)

    @staticmethod
    def _to_report(doc: dict) -> DailyReport:
        return DailyReport.model_validate(doc)

    def load_report(self, run_date: Date) -> DailyReport | None:
        doc = self.reports.find_one({"_id": run_date.isoformat()})
        return self._to_report(doc) if doc else None

    def latest_report(self) -> DailyReport | None:
        doc = self.reports.find_one(sort=[("run_date", -1)])
        return self._to_report(doc) if doc else None

    def list_runs(self) -> list[dict]:
        return [
            {"run_date": d["run_date"], "universe_size": d.get("universe_size"),
             "created_at": d.get("created_at")}
            for d in self.reports.find(sort=[("run_date", -1)])
        ]

    def _iter_reports_asc(self):
        return self.reports.find(sort=[("run_date", 1)])

    def ticker_history(self, ticker: str) -> list[dict]:
        ticker = ticker.upper()
        out: list[dict] = []
        for d in self._iter_reports_asc():
            for a in d.get("analyses", []):
                if a.get("ticker") == ticker:
                    m = a.get("metrics", {}) or {}
                    out.append({
                        "run_date": d["run_date"], "lean": a.get("lean"),
                        "confidence": a.get("confidence"), "score": a.get("score"),
                        "last_price": m.get("last_price"), "beta": m.get("beta"),
                        "alpha": m.get("alpha"), "sharpe": m.get("sharpe"),
                    })
        return out

    def all_calls(self) -> list[dict]:
        out: list[dict] = []
        for d in self._iter_reports_asc():
            for a in d.get("analyses", []):
                out.append({
                    "run_date": d["run_date"], "ticker": a.get("ticker"),
                    "lean": a.get("lean"), "confidence": a.get("confidence"),
                    "last_price": (a.get("metrics", {}) or {}).get("last_price"),
                })
        return out

    def top_picks(self, run_date: Date, limit: int = 10) -> list[dict]:
        d = self.reports.find_one({"_id": run_date.isoformat()})
        if not d:
            return []
        analyses = sorted(
            d.get("analyses", []), key=lambda a: a.get("score") or 0, reverse=True
        )[:limit]
        return [
            {"ticker": a["ticker"], "name": a.get("name"), "lean": a.get("lean"),
             "confidence": a.get("confidence"), "score": a.get("score")}
            for a in analyses
        ]

    # ---- outcomes ----------------------------------------------------------------------
    def upsert_outcome(
        self, run_date: str, ticker: str, horizon_days: int, lean: str,
        confidence: float | None, price_then: float | None, price_later: float | None,
        forward_return: float | None, correct: bool | None,
    ) -> None:
        key = {"run_date": run_date, "ticker": ticker, "horizon_days": horizon_days}
        self.outcomes.replace_one(
            key,
            {**key, "lean": lean, "confidence": confidence, "price_then": price_then,
             "price_later": price_later, "forward_return": forward_return,
             "correct": None if correct is None else int(correct)},
            upsert=True,
        )

    def outcomes_rows(self) -> list[dict]:
        return [
            {k: v for k, v in d.items() if k != "_id"}
            for d in self.outcomes.find({"forward_return": {"$ne": None}})
        ]
