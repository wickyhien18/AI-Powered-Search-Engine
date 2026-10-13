"""
config.py - Single source of truth for shared settings, loaded from .env.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# --- Web search (Tavily) ---
# Get a free key at https://tavily.com
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Tavily scores every result from 0 to 1. Results scoring below MIN_SCORE are
# dropped before they reach the LLM, so junk pages cannot leak into answers.
# Observed so far: irrelevant junk scored <= 0.022, relevant sources >= 0.477,
# weak but on-topic ones around 0.10. 0.05 is a deliberately conservative cut:
# raise it only after collecting more samples.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.05"))

# --- LLM (Groq) ---
# Get a free key at https://console.groq.com
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")

# --- CORS ---
# Comma-separated list of frontend origins allowed to call this API.
# The default keeps local development and the deployed Vercel frontend working.
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,https://ai-powered-search-engine-rosy.vercel.app",
    ).split(",")
    if origin.strip()
]

# --- Legacy settings (BBC corpus pipeline) ---
# NOT used by the web search flow in main.py. Kept only so the paused scripts
# (ingest.py, evaluate.py, generate_synthetic_queries.py) keep importing.
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = 384  # output dimension of bge-small-en-v1.5
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "bbc_news")
