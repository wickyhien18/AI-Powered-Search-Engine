"""
config.py — Single source of truth for shared settings, loaded from .env
Both ingest.py and main.py import from here instead of hardcoding values.
"""

import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")  # chat-capable model, used only for RAG answer generation
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "bbc_news")
EMBEDDING_DIM = 768  # fixed by nomic-embed-text's architecture, not something .env should override