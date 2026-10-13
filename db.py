"""
db.py - Conversation and message storage backed by SQLite.
This file knows nothing about search or LLMs; it is a plain read/write layer.
"""

import json
import sqlite3
from datetime import datetime, timezone

DB_PATH = "chat_history.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # access columns by name, e.g. row["title"]
    return conn


def init_db():
    """Create the tables if they do not exist. Safe to call on every startup."""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)
    conn.commit()
    conn.close()


def create_conversation(title: str) -> int:
    """Create a new conversation and return its id."""
    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO conversations (title, created_at) VALUES (?, ?)",
        (title, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


def add_message(conversation_id: int, role: str, content: str, sources: list | None = None):
    """Store one message (user or assistant) in an existing conversation.
    Sources are stored as a JSON string because SQLite has no native list/dict type."""
    conn = get_connection()
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
        (
            conversation_id,
            role,
            content,
            json.dumps(sources) if sources is not None else None,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def list_conversations() -> list[dict]:
    """All conversations for the sidebar, newest first."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, title, created_at FROM conversations ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_conversation_messages(conversation_id: int) -> list[dict]:
    """All messages of one conversation in chronological order.
    Pure database read: no search or LLM call is made."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT role, content, sources FROM messages WHERE conversation_id = ? ORDER BY id ASC",
        (conversation_id,),
    ).fetchall()
    conn.close()

    return [
        {
            "role": row["role"],
            "content": row["content"],
            "sources": json.loads(row["sources"]) if row["sources"] else None,
        }
        for row in rows
    ]
