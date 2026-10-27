# Wire Search

> A Perplexity-style answer engine: your question goes to live web search, an LLM writes the answer from the numbered sources, and every claim carries a clickable `[n]` citation that maps to a source URL. It started as a hybrid-retrieval RAG system over the BBC News archive and was rebuilt around web search as a hands-on AI Engineering portfolio project.

![License](https://img.shields.io/badge/license-MIT-blue)

**Live demo:** [AI-Powered Search Engine](https://ai-powered-search-engine-18bt.vercel.app/)

---

## Table of Contents

- [Introduction](#introduction)
- [Demo](#demo)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [How It Works](#how-it-works)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API](#api)
- [Project Structure](#project-structure)
- [Data & Evaluation](#data--evaluation)
- [Testing](#testing)
- [Deployment](#deployment)
- [Privacy](#privacy)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [References](#references)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## Introduction

**Wire Search** answers natural-language questions, including follow-ups that refer to earlier turns, using live web search instead of a fixed corpus. Each answer is written only from the retrieved sources, cites them with `[n]` markers, and the API returns the sources (title, URL, excerpt, relevance score) alongside the text.

The project went through two phases:

1. **v1: hybrid retrieval over a static archive.** 2,225 BBC News articles, dense (semantic) and sparse (BM25) retrieval fused with Reciprocal Rank Fusion in Qdrant, RAG answers, conversational memory, a hand-labeled evaluation set, deployed on free tiers.
2. **v2: web-grounded answers.** Tavily web search, relevance filtering calibrated on measured scores, an LLM that must cite its sources, query rewriting for follow-ups, rate limiting to protect the free-tier search quota, and an offline test suite.

## Demo

The interface uses a "newsroom wire" theme: a sidebar with example questions and past conversations, and a chat panel where each answer shows its `[n]` citations as links and a collapsible list of sources with their domains.

## Features

- Web-grounded answers: live search results are injected into the prompt, and the model must cite the numbered source for every claim
- Clickable citations: `[n]` in the answer links to the exact source; numbers that match no source stay as plain text
- Conversational memory: follow-ups are rewritten into standalone search queries that keep the specifics of what was discussed
- Relevance filtering: results below a score cutoff (0.20) are dropped before they reach the LLM; if nothing survives, the API says so instead of guessing
- Cleaner sources: Wikipedia reference-list lines are stripped from snippets so the model does not read article dates as event dates
- Rate limiting: per-IP and global daily limits in Redis protect the 1,000-credit free search quota
- Conversation persistence: conversations are stored in SQLite and reopen instantly without re-running any AI call (see [Privacy](#privacy))
- Offline test suite: search provider, LLM and Redis are faked, so tests run in seconds and cost nothing

## Tech Stack

| Component            | Technology                                                                     |
| -------------------- | ------------------------------------------------------------------------------ |
| Language             | Python (backend), TypeScript (frontend)                                        |
| Backend framework    | FastAPI + Uvicorn                                                              |
| Frontend framework   | Next.js (App Router) + react-markdown                                          |
| Web search           | Tavily API (basic depth), isolated behind `search.py`                          |
| LLM (generation)     | Groq API: `openai/gpt-oss-20b`, through `langchain-groq`                       |
| Orchestration        | LangChain (`langchain-core` messages, `langchain-groq`)                        |
| Rate limiting        | Upstash Redis (`redis-py`, one Lua script)                                     |
| Conversation storage | SQLite                                                                         |
| Tests / CI           | Offline test suite with fakes (`fakeredis`), GitHub Actions                    |
| Deployment           | Render (backend), Vercel (frontend), Upstash (Redis)                           |
| v1 only              | Qdrant Cloud, FastEmbed (`BAAI/bge-small-en-v1.5` dense, `Qdrant/bm25` sparse) |

## How It Works

```
Browser (Next.js, Vercel)
   |  POST /ask  {query, history, conversation_id}
   v
FastAPI (Render)
   1. rate limit check (Redis, 3 fixed windows)       -> 429 / 503
   2. follow-up question? rewrite it into a standalone
      search query using the chat history (Groq)
   3. Tavily search (basic depth)
   4. clean + filter: drop Wikipedia reference lines,
      drop results scoring below MIN_SCORE (0.20)
   5. number the sources [1..n]; Groq answers using only
      them, citing [n] after each claim
   6. store the conversation (SQLite), return answer + sources
```

If no source survives the filter, the API answers "no relevant sources" without calling the LLM.

## System Requirements

- Python 3.12 (the version tested in CI)
- Node.js >= 18
- No GPU required: the LLM runs on Groq's hosted infrastructure
- External services (all have free tiers): Tavily, Groq, Upstash Redis (optional locally)

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/wickyhien18/AI-Powered-Search-Engine.git
cd AI-Powered-Search-Engine

# 2. Backend: Python virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt   # runtime + test dependencies

# 3. Frontend
cd frontend
npm install
cd ..
```

## Configuration

Create a `.env` file in the project root (see `.env.example`):

```env
TAVILY_API_KEY=your-tavily-api-key
GROQ_API_KEY=your-groq-api-key

# Optional: defaults are set in config.py
LLM_MODEL=openai/gpt-oss-20b
MIN_SCORE=0.20
CORS_ORIGINS=http://localhost:3000

# Rate limiting: leave REDIS_URL empty to turn it off (local development)
REDIS_URL=
RATE_LIMIT_IP_PER_MINUTE=5
RATE_LIMIT_IP_PER_DAY=10
RATE_LIMIT_GLOBAL_PER_DAY=30
RATE_LIMIT_FAIL_OPEN=false
TRUSTED_PROXY_HOPS=1
RATE_LIMIT_DEBUG=false
```

| Variable                                                                         | Meaning                                                                                 |
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `MIN_SCORE`                                                                      | Relevance cutoff for Tavily results (0 to 1)                                            |
| `REDIS_URL`                                                                      | `rediss://` URL of the rate-limit Redis; empty turns rate limiting off                  |
| `RATE_LIMIT_IP_PER_MINUTE`, `RATE_LIMIT_IP_PER_DAY`, `RATE_LIMIT_GLOBAL_PER_DAY` | Request limits (0 disables one); the global limit counts per UTC day                    |
| `RATE_LIMIT_FAIL_OPEN`                                                           | `true` lets requests through when Redis is down; the default `false` returns 503        |
| `TRUSTED_PROXY_HOPS`                                                             | Reverse proxies that append to `X-Forwarded-For` (the client IP is read from the right) |
| `RATE_LIMIT_DEBUG`                                                               | Logs the forwarding headers so the real hop count can be measured                       |

For the frontend, create `frontend/.env`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

> The `NEXT_PUBLIC_` prefix is required: Next.js only exposes prefixed variables to browser-side code. See [Troubleshooting](#troubleshooting).

## Usage

**1. Run the backend API:**

```bash
uvicorn main:app --reload --port 8000
# API at http://localhost:8000, interactive docs at /docs
```

**2. Run the frontend (separate terminal):**

```bash
cd frontend
npm run dev
# App at http://localhost:3000
```

## API

| Endpoint                                        | What it does                                                                                                                                                                                                                                     |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `POST /ask`                                     | Full pipeline above. Body: `query`, `top_k` (1-10, default 5), `history` (list of `{role, content}`), `conversation_id` (optional). Returns `query`, `search_query`, `answer`, `sources` (`title`, `url`, `snippet`, `score`), `conversation_id` |
| `POST /search`                                  | Web search only, no LLM (useful to inspect what the search provider returns)                                                                                                                                                                     |
| `GET /conversations`, `GET /conversations/{id}` | Sidebar list and stored messages (database read only, not rate limited)                                                                                                                                                                          |
| `GET /health`                                   | Liveness check                                                                                                                                                                                                                                   |

Errors: `429` `{code: "rate_limited", limit, retry_after, message}` with a `Retry-After` header; `503` `{code: "rate_limiter_unavailable", message}`; `502` (detail is a string) when the search provider fails or the model returns an empty answer.

## Project Structure

```
.
├── main.py                 # FastAPI app: /ask, /search, /conversations; rewriter and answer prompts
├── search.py               # Tavily adapter: normalisation, reference-line stripping, score filter
├── rate_limit.py           # Redis rate limiter (Lua script) and client-IP logic
├── config.py               # Centralized settings, loaded from .env
├── db.py                   # SQLite persistence layer (conversations/messages)
├── test_backend.py         # Offline test suite (fake Tavily, Groq and Redis)
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # + test dependencies
├── probe_scores.py         # One-off measurement of the Tavily score scale
├── .env.example
├── .github/workflows/tests.yml   # CI: runs the offline tests
├── frontend/
│   ├── app/
│   │   ├── page.tsx        # Chat UI: sources, clickable citations, error banner
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── utils/
│   │   ├── api.ts          # askQuestion + error handling per status code
│   │   └── citations.ts    # [n] to link conversion, URL safety helpers
│   ├── config/env.ts
│   └── package.json
└── (v1, BBC pipeline)
    ├── setup.py            # Kaggle dataset download
    ├── ingest.py           # CSV -> chunk -> embed -> Qdrant
    ├── evaluate.py         # Golden-set evaluation (Recall@k, MRR, Precision@k)
    ├── label_helper.py     # Manual ground-truth labeling tool
    └── data/               # BBC News CSV (not committed)
```

## Data & Evaluation

### v1: retrieval over the BBC archive

- **Dataset:** [BBC Articles Cleaned](https://www.kaggle.com/datasets/dimasmunoz/bbc-articles-cleaned) (Kaggle): 2,225 articles, 5 categories (business, entertainment, politics, sport, tech)
- **Preprocessing:** text is already lowercased/stripped of punctuation in the source dataset; chunked with `RecursiveCharacterTextSplitter` (`chunk_size=500`, `chunk_overlap=50`)
- **Evaluation methodology:** a hand-labeled golden set (queries with manually verified correct `article_id`s, grouped into `proper_noun` / `paraphrase` / `general` categories) is used to compute Recall@5, MRR, and Precision@5 for dense-only and hybrid retrieval

**Confirmed result (initial 2-query comparison):**

| Method                  | Recall@5 | MRR   | Precision@5 |
| ----------------------- | -------- | ----- | ----------- |
| Dense-only              | 1.00     | 1.000 | 0.80        |
| Hybrid (dense + sparse) | 1.00     | 1.000 | 0.80        |

Aggregate scores were tied, but per-query results were not: hybrid retrieval scored noticeably higher on a proper-noun query (an artist's surname), the case hybrid search is specifically meant to help with, while dense-only scored higher on a general semantic query. **The honest takeaway: hybrid search is not a universal improvement over dense-only on this dataset. It helps most on exact-term/proper-noun queries, and a larger, category-balanced golden set is needed before drawing a confident overall conclusion.**

### v2: web-grounded answers

- **Relevance cutoff from data.** Over 82 measured search results, keyword-matching junk pages scored at most 0.0589 and relevant ones at least 0.3233, with nothing in between, so the cutoff is 0.20. A score measures how well a page matches the query text, not the user's intent: ambiguous queries such as `first` or `tell me more about it` return pages scoring 0.39-0.76, so the cutoff cannot catch them. Query rewriting quality matters more than the threshold.
- **Answers audited by hand.** There is no automated answer-quality metric yet. Answers were checked against their cited snippets, which surfaced the failure classes listed under [Known Limitations](#known-limitations) and led to the rules in the answer prompt (cite only the source that states a claim, report disagreeing values, keep dates and hedging words, no generalizing from examples).
- **Cost awareness.** Measured: every `/ask` or `/search` costs one Tavily credit, independent of `top_k`; the free tier has 1,000 credits per month, so the default global cap of 30 requests per UTC day (930 per month) stays under it.

## Testing

```bash
python test_backend.py    # offline test suite: no API keys, no credits
python evaluate.py        # v1 retrieval evaluation (needs Qdrant and the extra v1 packages below)
```

The offline suite fakes Tavily, Groq and Redis (`fakeredis`, which runs the real Lua script), and covers query rewriting, source numbering, persistence, error handling, CORS, the Tavily adapter (score filter, reference-line stripping), and the rate limiter (per-minute window, global cap, rejected requests costing no quota, client-IP parsing, fail-closed behaviour). It runs on every push and pull request through GitHub Actions.

The v1 scripts (`setup.py`, `ingest.py`, `evaluate.py`, `label_helper.py`) need packages that are not in `requirements.txt`: `langchain-community`, `langchain-qdrant`, `langchain-text-splitters`, `langchain-ollama`, `qdrant-client`, `fastembed`, `pandas` and `kaggle`.

## Deployment

Deployed on free tiers:

1. **Backend (Render):** free web service; start command `uvicorn main:app --host 0.0.0.0 --port $PORT`. Set `TAVILY_API_KEY`, `GROQ_API_KEY`, `REDIS_URL` and `CORS_ORIGINS` (the exact Vercel production origin, without a trailing slash). Set `RATE_LIMIT_DEBUG=true` once to measure `TRUSTED_PROXY_HOPS`, then turn it off.
2. **Redis (Upstash):** a free Redis database; copy its `rediss://` URL into `REDIS_URL`.
3. **Frontend (Vercel):** free tier; set `NEXT_PUBLIC_API_URL` to the Render backend URL and **redeploy** (the value is baked in at build time).
4. An external uptime monitor pings the backend periodically to reduce free-tier cold starts.

## Privacy

The demo stores conversations in one shared SQLite database. The sidebar lists every conversation from every visitor, and any conversation can be opened by its id. Do not enter personal or sensitive information. SQLite lives on Render's free-tier filesystem, so conversations can disappear after a restart or redeploy.

## Known Limitations

- Citations are not machine-verified. Observed failures: a claim cited to the wrong source, a claim stated more broadly than its source ("all" from a few examples), a statement attributed to the wrong speaker, and relative dates copied from an old article.
- Reference-line stripping handles two Wikipedia footnote formats; other formats pass through.
- Social-media pages can be returned and cited; there is no source-quality filter.
- The same article can appear under several URLs; sources are not deduplicated.
- Search snippets vary with the query wording, so the same page can yield different excerpts.
- The model is a reasoning model: if its token budget runs out it can return an empty answer, which the API reports as `502` instead of storing it.
- Groq errors (rate limit, timeout) are not caught yet and surface as a generic server error.
- If the Redis quota runs out, the API fails closed and returns `503` to everyone.

## Troubleshooting

<details>
<summary>Every visitor gets <code>429</code> almost immediately on Render</summary>

The per-IP limits are counting the proxy's address instead of the visitor's, so everyone shares one counter. Set `RATE_LIMIT_DEBUG=true`, send a request from a device whose public IP you know (for example a phone on mobile data), read the `x-forwarded-for` value in the Render logs, and set `TRUSTED_PROXY_HOPS` so the N-th entry from the end is that IP.

</details>

<details>
<summary>Requests fail with <code>503</code> and <code>rate_limiter_unavailable</code></summary>

Redis is unreachable or out of quota, and the limiter fails closed on purpose to protect the search credits. Check `REDIS_URL` and the Upstash dashboard. Setting `RATE_LIMIT_FAIL_OPEN=true` lets requests through while Redis is down, at the cost of an unprotected quota.

</details>

<details>
<summary>The browser reports a CORS error</summary>

`CORS_ORIGINS` must contain the exact origin of the frontend (scheme and host, no trailing slash). Vercel preview deployments have different origins from the production domain.

</details>

<details>
<summary>Frontend calls <code>localhost:8000</code> even after deploying (<code>ERR_CONNECTION_REFUSED</code>)</summary>

The API URL environment variable is missing the `NEXT_PUBLIC_` prefix. Next.js only bundles prefixed variables into client-side code: an unprefixed variable is `undefined` in the browser and silently falls back to its default. Rename it to `NEXT_PUBLIC_API_URL`, set it on Vercel, and **redeploy** (the value is baked in at build time).

</details>

<details>
<summary>Groq API returns 404 "model does not exist or you do not have access to it"</summary>

Groq's available model list changes over time, and some models move to enterprise-only access. Query your account's actual available models before assuming a model name still works:

```bash
curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"
```

</details>

<details>
<summary>LLM returns an empty <code>answer</code> with no error</summary>

Reasoning-capable models (e.g. `openai/gpt-oss-20b`) spend part of their token budget on internal reasoning (exposed separately as `reasoning_content`) before writing the final answer: both count against `max_tokens`. If the limit is too low, the model can exhaust its budget mid-thought and return empty final content. Increase `max_tokens`.

</details>

<details>
<summary>v1 only: <code>QdrantVectorStoreError</code> "Existing collection is built with unnamed dense vector"</summary>

The collection was created with a different embedding schema (e.g. before switching embedding models, which changes vector dimensions). Delete the collection and re-run `ingest.py`:

```bash
curl -X DELETE "$QDRANT_URL/collections/bbc_news" -H "api-key: $QDRANT_API_KEY"
python ingest.py
```

</details>

<details>
<summary>v1 only: <code>ResponseHandlingException: The write operation timed out</code> during ingest</summary>

Happens when writing to a remote Qdrant Cloud cluster (vs. localhost): the default client timeout is too short for real network latency. Increase `timeout=` when creating `QdrantClient`, and/or reduce the ingest batch size.

</details>

## Contributing

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit using [Conventional Commits](https://www.conventionalcommits.org/) (e.g. `feat: add streaming responses`)
4. Make sure `python test_backend.py` passes, then push and open a Pull Request describing the change and why

## References

- [Tavily documentation](https://docs.tavily.com/)
- [Groq documentation](https://console.groq.com/docs)
- [Upstash Redis documentation](https://upstash.com/docs/redis)
- [LangChain documentation](https://python.langchain.com/)
- [BBC Articles Cleaned dataset](https://www.kaggle.com/datasets/dimasmunoz/bbc-articles-cleaned) (Kaggle, v1)
- [Qdrant documentation](https://qdrant.tech/documentation/) (v1)

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

**Giap Minh Hien** — [WickyHien](https://github.com/wickyhien18)

## Acknowledgments

- [dimasmunoz](https://www.kaggle.com/dimasmunoz) for the cleaned BBC Articles dataset
- Tavily, Groq, Upstash, LangChain, Qdrant and FastEmbed for the open-source and free-tier tooling this project builds on
