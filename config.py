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

# Off by default — the cross-encoder rerank was tested and found UNRELIABLE on
# this dataset specifically, because the stored chunk text has been stripped of
# punctuation/casing during ingest (see Giai đoạn 3), which doesn't match the
# natural-language text the rerank model was trained on. Code is kept intact,
# ready to re-enable once ingest.py is updated to preserve the original text
# for reranking purposes (planned follow-up, not done yet).
ENABLE_RERANK = os.getenv("ENABLE_RERANK", "false").lower() == "true"