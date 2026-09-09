from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class MemoryDatabase:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event TEXT NOT NULL,
                    context TEXT NOT NULL,
                    result TEXT NOT NULL,
                    importance REAL NOT NULL,
                    confidence REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS semantic_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    information TEXT NOT NULL,
                    source TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    tags TEXT NOT NULL,
                    relationships TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    initial_state TEXT NOT NULL,
                    evidence_used TEXT NOT NULL,
                    information_value TEXT NOT NULL DEFAULT '{}',
                    hypotheses TEXT NOT NULL,
                    predictions TEXT NOT NULL,
                    considered_actions TEXT NOT NULL,
                    selected_action TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    final_result TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS resources (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    state TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    provider TEXT NOT NULL DEFAULT '',
                    version TEXT,
                    location TEXT,
                    discovery_source TEXT,
                    acquisition_method TEXT,
                    provenance TEXT NOT NULL DEFAULT '[]',
                    acquisition_history TEXT NOT NULL DEFAULT '[]',
                    verification_results TEXT NOT NULL DEFAULT '[]',
                    limitations TEXT NOT NULL DEFAULT '[]',
                    confidence REAL NOT NULL DEFAULT 0,
                    last_verified_at TEXT,
                    last_verified_status TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS capabilities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    type TEXT NOT NULL,
                    state TEXT NOT NULL,
                    inputs TEXT NOT NULL DEFAULT '[]',
                    outputs TEXT NOT NULL DEFAULT '[]',
                    prerequisites TEXT NOT NULL DEFAULT '[]',
                    effects TEXT NOT NULL DEFAULT '[]',
                    risk_level TEXT NOT NULL,
                    authorization_required INTEGER NOT NULL DEFAULT 0,
                    reversibility TEXT NOT NULL DEFAULT 'unknown',
                    verification_method TEXT NOT NULL DEFAULT '',
                    provenance TEXT NOT NULL DEFAULT '[]',
                    learned_uses TEXT NOT NULL DEFAULT '[]',
                    limitations TEXT NOT NULL DEFAULT '[]',
                    confidence REAL NOT NULL DEFAULT 0,
                    last_verified_at TEXT,
                    last_verified_status TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS resource_capabilities (
                    resource_id TEXT NOT NULL,
                    capability_id TEXT NOT NULL,
                    provenance TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (resource_id, capability_id)
                );

                CREATE TABLE IF NOT EXISTS capability_compositions (
                    capability_id TEXT PRIMARY KEY,
                    required_capabilities TEXT NOT NULL,
                    description TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS capability_acquisition_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    option_id TEXT NOT NULL,
                    capability_id TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result TEXT NOT NULL,
                    state_before TEXT NOT NULL,
                    state_after TEXT NOT NULL,
                    verification_result TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS authorization_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    requested_action TEXT NOT NULL,
                    risk_classification TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    requested_scope TEXT NOT NULL,
                    authorization_required INTEGER NOT NULL,
                    approved INTEGER NOT NULL,
                    result_reason TEXT NOT NULL,
                    resulting_execution TEXT NOT NULL DEFAULT '',
                    verification_result TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS objective_models (
                    objective_id TEXT PRIMARY KEY,
                    objective_text TEXT NOT NULL,
                    desired_outcome TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    objective_id TEXT NOT NULL,
                    parent_task_id TEXT,
                    status TEXT NOT NULL,
                    title TEXT NOT NULL,
                    priority REAL NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS engine_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    objective_id TEXT NOT NULL,
                    objective_text TEXT NOT NULL,
                    status TEXT NOT NULL,
                    cycles INTEGER NOT NULL,
                    result TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                """
            )
            columns = {
                row["name"]
                for row in db.execute("PRAGMA table_info(analysis_history)").fetchall()
            }
            if "information_value" not in columns:
                db.execute("ALTER TABLE analysis_history ADD COLUMN information_value TEXT NOT NULL DEFAULT '{}'")

    @staticmethod
    def dumps(value: Any) -> str:
        return json.dumps(value, ensure_ascii=True, default=str)
