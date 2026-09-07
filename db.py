"""
db.py — Lưu trữ conversation/message bằng SQLite.
Tách riêng khỏi main.py: file này KHÔNG biết gì về Qdrant/Ollama/LLM,
chỉ thuần là lớp đọc/ghi dữ liệu.
"""

import sqlite3
import json
from datetime import datetime, timezone

DB_PATH = "chat_history.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # cho phép truy cập cột theo tên, vd row["title"]
    return conn


def init_db():
    """Tạo bảng nếu chưa tồn tại — an toàn để gọi mỗi lần server khởi động."""
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
    """Tạo 1 conversation mới, trả về id vừa tạo."""
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
    """Lưu 1 tin nhắn (user hoặc assistant) vào 1 conversation đã tồn tại.
    sources được lưu dạng JSON string vì SQLite không có kiểu list/dict native."""
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
    """Danh sách conversation cho sidebar — mới nhất lên đầu."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, title, created_at FROM conversations ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_conversation_messages(conversation_id: int) -> list[dict]:
    """Toàn bộ tin nhắn của 1 conversation, theo đúng thứ tự thời gian —
    đây là dữ liệu được LOAD THẲNG, không chạy lại AI gì cả."""
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
