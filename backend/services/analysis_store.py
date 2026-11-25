from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parents[2] / "storage" / "app" / "deepshield.sqlite3"


class AnalysisStore:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or DB_PATH)

    @contextmanager
    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("""CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL, media_type TEXT NOT NULL,
            filename TEXT NOT NULL, sha256 TEXT NOT NULL, metadata_json TEXT NOT NULL,
            result_json TEXT NOT NULL, status TEXT NOT NULL
        )""")
        connection.commit()
        try:
            yield connection
        finally:
            connection.close()

    def add(self, *, media_type: str, filename: str, sha256: str, metadata: dict[str, Any], result: dict[str, Any], status: str = "complete") -> dict[str, Any]:
        record = {"id": str(uuid.uuid4()), "created_at": datetime.now(timezone.utc).isoformat(), "media_type": media_type, "filename": filename, "sha256": sha256, "metadata": metadata, "result": result, "status": status}
        with self._connect() as db:
            db.execute("INSERT INTO analyses VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (record["id"], record["created_at"], media_type, filename, sha256, json.dumps(metadata), json.dumps(result), status))
            db.commit()
        return record

    def list(self, *, query: str = "", media_type: str = "", limit: int = 100) -> list[dict[str, Any]]:
        clauses=[]; params=[]
        if query:
            clauses.append("(filename LIKE ? OR id LIKE ?)"); params.extend([f"%{query}%",f"%{query}%"])
        if media_type:
            clauses.append("media_type = ?"); params.append(media_type)
        where=(" WHERE "+" AND ".join(clauses)) if clauses else ""
        with self._connect() as db:
            rows=db.execute("SELECT * FROM analyses"+where+" ORDER BY created_at DESC LIMIT ?",[*params,max(1,min(limit,500))]).fetchall()
        return [self._decode(row) for row in rows]

    def get(self, analysis_id: str) -> dict[str, Any] | None:
        with self._connect() as db: row=db.execute("SELECT * FROM analyses WHERE id = ?",(analysis_id,)).fetchone()
        return self._decode(row) if row else None

    @staticmethod
    def _decode(row):
        return {"id":row["id"],"created_at":row["created_at"],"media_type":row["media_type"],"filename":row["filename"],"sha256":row["sha256"],"metadata":json.loads(row["metadata_json"]),"result":json.loads(row["result_json"]),"status":row["status"]}

    def counts(self) -> dict[str, int]:
        with self._connect() as db: rows=db.execute("SELECT media_type, COUNT(*) AS n FROM analyses GROUP BY media_type").fetchall()
        counts={"image":0,"video":0}
        counts.update({r["media_type"]:r["n"] for r in rows})
        return counts
