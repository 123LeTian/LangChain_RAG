"""SQLite-backed chat storage."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import RLock
from typing import Any, List, Optional, Union

from src.chat.schemas import ChatMessage, ChatPreset, ChatSession, utc_now
from src.chat_storage.chat_store import UNSET
from src.config.runtime_config import get_chat_db_path


class SQLiteChatStore:
    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path) if db_path is not None else get_chat_db_path()
        self._lock = RLock()
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            with self._connect() as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        model_id TEXT,
                        preset_id TEXT,
                        rag_mode TEXT,
                        knowledge_base_id TEXT,
                        total_prompt_tokens INTEGER NOT NULL DEFAULT 0,
                        total_completion_tokens INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                        content TEXT NOT NULL,
                        citations TEXT,
                        trace TEXT,
                        prompt_tokens INTEGER NOT NULL DEFAULT 0,
                        completion_tokens INTEGER NOT NULL DEFAULT 0,
                        latency_ms INTEGER,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (session_id)
                            REFERENCES chat_sessions(id)
                            ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated_at
                        ON chat_sessions(updated_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
                        ON chat_messages(session_id, created_at ASC);

                    CREATE TABLE IF NOT EXISTS chat_presets (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT NOT NULL DEFAULT '',
                        category TEXT NOT NULL DEFAULT '通用',
                        system_prompt TEXT NOT NULL,
                        rag_prompt_hint TEXT,
                        owner_type TEXT NOT NULL CHECK (owner_type IN ('user')),
                        is_default INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_chat_presets_updated_at
                        ON chat_presets(updated_at DESC);
                    """
                )
            # migration: add category column for existing databases
            try:
                with self._lock, self._connect() as conn:
                    conn.execute(
                        "ALTER TABLE chat_presets ADD COLUMN category TEXT NOT NULL DEFAULT '通用'"
                    )
            except Exception:
                pass
            self._initialized = True

    def create_session(self, session: ChatSession) -> ChatSession:
        self.initialize()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_sessions (
                    id, title, model_id, preset_id, rag_mode, knowledge_base_id,
                    total_prompt_tokens, total_completion_tokens, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    session.title,
                    session.model_id,
                    session.preset_id,
                    session.rag_mode,
                    session.knowledge_base_id,
                    session.total_prompt_tokens,
                    session.total_completion_tokens,
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                ),
            )
        return session

    def list_sessions(self) -> List[ChatSession]:
        self.initialize()
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chat_sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [self._row_to_session(row) for row in rows]

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        self.initialize()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM chat_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        return self._row_to_session(row) if row is not None else None

    def update_session(
        self,
        session_id: str,
        *,
        title: Any = UNSET,
        model_id: Any = UNSET,
        preset_id: Any = UNSET,
        rag_mode: Any = UNSET,
        knowledge_base_id: Any = UNSET,
    ) -> Optional[ChatSession]:
        self.initialize()
        now = utc_now().isoformat()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM chat_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None

            conn.execute(
                """
                UPDATE chat_sessions
                SET title = ?,
                    model_id = ?,
                    preset_id = ?,
                    rag_mode = ?,
                    knowledge_base_id = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    row["title"] if title is UNSET else title,
                    row["model_id"] if model_id is UNSET else model_id,
                    row["preset_id"] if preset_id is UNSET else preset_id,
                    row["rag_mode"] if rag_mode is UNSET else rag_mode,
                    row["knowledge_base_id"] if knowledge_base_id is UNSET else knowledge_base_id,
                    now,
                    session_id,
                ),
            )
            updated = conn.execute(
                "SELECT * FROM chat_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        return self._row_to_session(updated)

    def delete_session(self, session_id: str) -> bool:
        self.initialize()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM chat_sessions WHERE id = ?",
                (session_id,),
            )
            return cursor.rowcount > 0

    def create_message(self, message: ChatMessage) -> ChatMessage:
        self.initialize()
        now = utc_now().isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages (
                    id, session_id, role, content, citations, trace,
                    prompt_tokens, completion_tokens, latency_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    message.session_id,
                    message.role,
                    message.content,
                    self._dump_json(message.citations),
                    self._dump_json(message.trace),
                    message.prompt_tokens,
                    message.completion_tokens,
                    message.latency_ms,
                    message.created_at.isoformat(),
                ),
            )
            conn.execute(
                """
                UPDATE chat_sessions
                SET total_prompt_tokens = total_prompt_tokens + ?,
                    total_completion_tokens = total_completion_tokens + ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    message.prompt_tokens,
                    message.completion_tokens,
                    now,
                    message.session_id,
                ),
            )
        return message

    def list_messages(self, session_id: str) -> List[ChatMessage]:
        self.initialize()
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM chat_messages
                WHERE session_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (session_id,),
            ).fetchall()
        return [self._row_to_message(row) for row in rows]

    def create_preset(self, preset: ChatPreset) -> ChatPreset:
        self.initialize()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_presets (
                    id, name, description, category, system_prompt, rag_prompt_hint,
                    owner_type, is_default, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    preset.id,
                    preset.name,
                    preset.description,
                    getattr(preset, "category", "通用") or "通用",
                    preset.system_prompt,
                    preset.rag_prompt_hint,
                    preset.owner_type,
                    1 if preset.is_default else 0,
                    (preset.created_at or utc_now()).isoformat(),
                    (preset.updated_at or utc_now()).isoformat(),
                ),
            )
        return preset

    def list_user_presets(self) -> List[ChatPreset]:
        self.initialize()
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM chat_presets
                WHERE owner_type = 'user'
                ORDER BY updated_at DESC
                """
            ).fetchall()
        return [self._row_to_preset(row) for row in rows]

    def get_user_preset(self, preset_id: str) -> Optional[ChatPreset]:
        self.initialize()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM chat_presets
                WHERE id = ? AND owner_type = 'user'
                """,
                (preset_id,),
            ).fetchone()
        return self._row_to_preset(row) if row is not None else None

    def update_user_preset(
        self,
        preset_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
        system_prompt: Optional[str] = None,
        rag_prompt_hint: Optional[str] = None,
    ) -> Optional[ChatPreset]:
        self.initialize()
        now = utc_now().isoformat()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM chat_presets
                WHERE id = ? AND owner_type = 'user'
                """,
                (preset_id,),
            ).fetchone()
            if row is None:
                return None

            # safely read category from row (may not exist in old databases)
            def _row_col(row_obj: Any, col: str, default: str = "") -> str:
                try:
                    return row_obj[col]
                except (IndexError, KeyError):
                    return default

            conn.execute(
                """
                UPDATE chat_presets
                SET name = ?, description = ?, category = ?, system_prompt = ?,
                    rag_prompt_hint = ?, updated_at = ?
                WHERE id = ? AND owner_type = 'user'
                """,
                (
                    name if name is not None else row["name"],
                    description if description is not None else row["description"],
                    category if category is not None else _row_col(row, "category", "通用"),
                    system_prompt if system_prompt is not None else row["system_prompt"],
                    rag_prompt_hint if rag_prompt_hint is not None else row["rag_prompt_hint"],
                    now,
                    preset_id,
                ),
            )
            updated = conn.execute(
                "SELECT * FROM chat_presets WHERE id = ?",
                (preset_id,),
            ).fetchone()
        return self._row_to_preset(updated)

    def delete_user_preset(self, preset_id: str) -> bool:
        self.initialize()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM chat_presets
                WHERE id = ? AND owner_type = 'user'
                """,
                (preset_id,),
            )
            return cursor.rowcount > 0

    def replace_session_preset(self, preset_id: str, replacement_preset_id: str) -> None:
        self.initialize()
        now = utc_now().isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE chat_sessions
                SET preset_id = ?, updated_at = ?
                WHERE preset_id = ?
                """,
                (replacement_preset_id, now, preset_id),
            )

    def search_messages(
        self,
        query: str,
        *,
        session_id: Optional[str] = None,
        role: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[List[dict[str, Any]], int]:
        self.initialize()
        pattern = f"%{query}%"
        where = ["m.content LIKE ?"]
        params: list[Any] = [pattern]
        if session_id:
            where.append("m.session_id = ?")
            params.append(session_id)
        if role:
            where.append("m.role = ?")
            params.append(role)
        where_sql = " AND ".join(where)

        with self._lock, self._connect() as conn:
            total = conn.execute(
                f"""
                SELECT COUNT(*)
                FROM chat_messages m
                JOIN chat_sessions s ON s.id = m.session_id
                WHERE {where_sql}
                """,
                tuple(params),
            ).fetchone()[0]
            rows = conn.execute(
                f"""
                SELECT
                    m.id AS message_id,
                    m.session_id,
                    s.title AS session_title,
                    m.role,
                    m.content,
                    m.created_at
                FROM chat_messages m
                JOIN chat_sessions s ON s.id = m.session_id
                WHERE {where_sql}
                ORDER BY m.created_at DESC, m.id DESC
                LIMIT ? OFFSET ?
                """,
                tuple([*params, limit, offset]),
            ).fetchall()

        return [dict(row) for row in rows], int(total)

    def global_chat_stats(self) -> dict[str, int]:
        self.initialize()
        with self._lock, self._connect() as conn:
            sessions_count = conn.execute(
                "SELECT COUNT(*) FROM chat_sessions"
            ).fetchone()[0]
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS messages_count,
                    COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0) AS completion_tokens
                FROM chat_messages
                """
            ).fetchone()
        return {
            "sessions_count": int(sessions_count),
            "messages_count": int(row["messages_count"]),
            "prompt_tokens": int(row["prompt_tokens"]),
            "completion_tokens": int(row["completion_tokens"]),
        }

    def session_chat_stats(self, session_id: str) -> Optional[dict[str, int]]:
        self.initialize()
        with self._lock, self._connect() as conn:
            session_exists = conn.execute(
                "SELECT 1 FROM chat_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if session_exists is None:
                return None
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS messages_count,
                    COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0) AS completion_tokens
                FROM chat_messages
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        return {
            "messages_count": int(row["messages_count"]),
            "prompt_tokens": int(row["prompt_tokens"]),
            "completion_tokens": int(row["completion_tokens"]),
        }

    def _row_to_session(self, row: sqlite3.Row) -> ChatSession:
        return ChatSession(
            id=row["id"],
            title=row["title"],
            model_id=row["model_id"],
            preset_id=row["preset_id"],
            rag_mode=row["rag_mode"],
            knowledge_base_id=row["knowledge_base_id"],
            total_prompt_tokens=row["total_prompt_tokens"],
            total_completion_tokens=row["total_completion_tokens"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_message(self, row: sqlite3.Row) -> ChatMessage:
        return ChatMessage(
            id=row["id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            citations=self._load_json(row["citations"]),
            trace=self._load_json(row["trace"]),
            prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            latency_ms=row["latency_ms"],
            created_at=row["created_at"],
        )

    def _row_to_preset(self, row: sqlite3.Row) -> ChatPreset:
        def _col(key: str, default: str = "") -> str:
            try:
                return row[key]
            except (IndexError, KeyError):
                return default
        return ChatPreset(
            id=row["id"],
            name=row["name"],
            description=_col("description", ""),
            category=_col("category", "通用"),
            system_prompt=row["system_prompt"],
            rag_prompt_hint=_col("rag_prompt_hint", None),
            owner_type=row["owner_type"],
            is_default=bool(row["is_default"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _dump_json(self, value: Optional[List[Any]]) -> Optional[str]:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False)

    def _load_json(self, value: Optional[str]) -> Optional[List[Any]]:
        if value is None:
            return None
        return json.loads(value)
