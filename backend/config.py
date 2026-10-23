"""
config.py - Single source of truth for shared settings, loaded from .env.
"""

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# Search for .env in current directory, backend directory, or project root
_env_path = find_dotenv(usecwd=True)
if not _env_path:
    _this_dir = Path(__file__).resolve().parent
    for _candidate in [_this_dir / ".env", _this_dir.parent / ".env"]:
        if _candidate.exists():
            _env_path = str(_candidate)
            break

load_dotenv(dotenv_path=_env_path)

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

# --- Rate limiting (Upstash Redis) ---
# Every /ask and /search request costs one Tavily credit, so the number of
# requests is capped. Leave REDIS_URL empty to turn rate limiting off
# (local development and the offline tests do this).
# Upstash console -> your database -> "Redis" tab -> copy the rediss:// URL.
REDIS_URL = os.getenv("REDIS_URL")
RATE_LIMIT_IP_PER_MINUTE = int(os.getenv("RATE_LIMIT_IP_PER_MINUTE", "5"))
RATE_LIMIT_IP_PER_DAY = int(os.getenv("RATE_LIMIT_IP_PER_DAY", "10"))
RATE_LIMIT_GLOBAL_PER_DAY = int(os.getenv("RATE_LIMIT_GLOBAL_PER_DAY", "30"))  # 30 x 31 days = 930 < 1000 credits
# If Redis is unreachable: "true" lets requests through (credits unprotected),
# "false" rejects them with 503 (credits protected, site unavailable).
RATE_LIMIT_FAIL_OPEN = os.getenv("RATE_LIMIT_FAIL_OPEN", "false").lower() == "true"
# Number of reverse proxies between the internet and this app that append to
# X-Forwarded-For. Measure it on the real deployment with RATE_LIMIT_DEBUG=true.
TRUSTED_PROXY_HOPS = int(os.getenv("TRUSTED_PROXY_HOPS", "1"))
RATE_LIMIT_DEBUG = os.getenv("RATE_LIMIT_DEBUG", "false").lower() == "true"

# --- CORS ---
# Comma-separated list of frontend origins allowed to call this API.
# The default keeps local development and the deployed Vercel frontend working.
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000",
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