"""SQLite persistence for the dataset, review queue, and eval runs."""

from __future__ import annotations

import json
import sqlite3
import time


class DatasetStore:
    def __init__(self, path: str = "evalgen.db"):
        self.db = sqlite3.connect(path)
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS cases (
            case_id TEXT PRIMARY KEY, payload TEXT, status TEXT,
            added_at REAL);
        CREATE TABLE IF NOT EXISTS reviews (
            case_id TEXT PRIMARY KEY, decision TEXT, reviewer TEXT,
            edited_payload TEXT, at REAL);
        CREATE TABLE IF NOT EXISTS eval_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT, model TEXT,
            report TEXT, at REAL);
        """)

    def add_cases(self, cases: list[dict], status: str) -> None:
        for case in cases:
            self.db.execute(
                "INSERT OR IGNORE INTO cases VALUES (?, ?, ?, ?)",
                (case["case_id"], json.dumps(case), status, time.time()))
        self.db.commit()

    def dataset(self, status: str = "active") -> list[dict]:
        rows = self.db.execute(
            "SELECT payload FROM cases WHERE status = ?", (status,))
        return [json.loads(row[0]) for row in rows]

    def review_queue(self) -> list[dict]:
        return self.dataset("pending_review")

    def resolve_review(self, case_id: str, decision: str,
                       reviewer: str, edited: dict | None = None) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO reviews VALUES (?, ?, ?, ?, ?)",
            (case_id, decision, reviewer,
             json.dumps(edited) if edited else None, time.time()))
        if decision == "approve":
            self.db.execute(
                "UPDATE cases SET status = 'active' WHERE case_id = ?",
                (case_id,))
        elif decision == "edit" and edited:
            self.db.execute(
                "UPDATE cases SET status = 'active', payload = ?"
                " WHERE case_id = ?", (json.dumps(edited), case_id))
        else:
            self.db.execute(
                "UPDATE cases SET status = 'rejected' WHERE case_id = ?",
                (case_id,))
        self.db.commit()

    def save_run(self, model: str, report: dict) -> int:
        cursor = self.db.execute(
            "INSERT INTO eval_runs (model, report, at) VALUES (?, ?, ?)",
            (model, json.dumps(report), time.time()))
        self.db.commit()
        return cursor.lastrowid

    def last_runs(self, n: int = 2) -> list[dict]:
        rows = self.db.execute(
            "SELECT run_id, model, report FROM eval_runs"
            " ORDER BY run_id DESC LIMIT ?", (n,))
        return [{"run_id": r[0], "model": r[1],
                 "report": json.loads(r[2])} for r in rows]

    def growth_stats(self) -> dict:
        by_status = dict(self.db.execute(
            "SELECT status, COUNT(*) FROM cases GROUP BY status"))
        cases = self.dataset()
        by_category: dict[str, int] = {}
        by_difficulty: dict[str, int] = {}
        for case in cases:
            by_category[case["category"]] = \
                by_category.get(case["category"], 0) + 1
            by_difficulty[case["difficulty"]] = \
                by_difficulty.get(case["difficulty"], 0) + 1
        return {"by_status": by_status, "by_category": by_category,
                "by_difficulty": by_difficulty,
                "active_total": len(cases)}
