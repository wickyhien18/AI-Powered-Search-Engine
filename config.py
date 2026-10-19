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
# Observed over 82 real results (45 earlier, 37 re-measured with
# probe_scores.py): junk scored <= 0.0589, relevant sources >= 0.3233, and
# nothing was seen in between. 0.20 sits inside that gap and was chosen to
# favor precision. Scores are not comparable across queries, so do not push
# this much higher: a query whose relevant results all score ~0.35 would
# lose every source.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.20"))

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