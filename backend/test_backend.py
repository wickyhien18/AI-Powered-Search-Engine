"""
test_backend.py - Offline tests for the web-search backend.

Tavily and Groq are replaced with fakes, so this makes NO network calls and
spends NO API credits. It checks the pipeline logic: query rewriting, source
numbering in the prompt, persistence, error handling and CORS.

Run: python test_backend.py
"""

import os
import sys
import tempfile
from pathlib import Path

# Add backend directory to sys.path so direct imports work regardless of working directory
_backend_dir = str(Path(__file__).resolve().parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

# Dummy keys so the real clients can be constructed at import time.
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")
os.environ["REDIS_URL"] = ""

import db

# Point the database at a throwaway file BEFORE main is imported.
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp.close()
db.DB_PATH = _tmp.name

import main
from fastapi.testclient import TestClient
from search import WebSearchError

client = TestClient(main.app)

FAKE_SOURCES = [
    {"title": "Title One", "url": "https://example.com/1", "snippet": "First excerpt [...] more.", "score": 0.9},
    {"title": "Title Two", "url": "https://example.com/2", "snippet": "Second excerpt.", "score": 0.7},
]


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Stands in for ChatGroq. A string payload is the query-rewrite call,
    a list payload is the answer call."""

    def __init__(self):
        self.calls = []
        self.answer = "Answer text [1][2]."
        self.rewrite = "rewritten standalone query"

    def invoke(self, payload):
        self.calls.append(payload)
        return FakeResponse(self.rewrite if isinstance(payload, str) else self.answer)


class FakeSearch:
    def __init__(self):
        self.calls = []
        self.results = FAKE_SOURCES
        self.error = None

    def __call__(self, query, max_results=5):
        self.calls.append({"query": query, "max_results": max_results})
        if self.error:
            raise self.error
        return self.results


fake_llm = FakeLLM()
fake_search = FakeSearch()
main.llm = fake_llm
main.web_search = fake_search


def reset():
    fake_llm.calls.clear()
    fake_llm.answer = "Answer text [1][2]."
    fake_search.calls.clear()
    fake_search.results = FAKE_SOURCES
    fake_search.error = None


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_first_turn_numbers_sources_and_skips_rewrite():
    reset()
    r = client.post("/ask", json={"query": "why are musicians protesting", "top_k": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == "Answer text [1][2]."
    assert body["sources"] == FAKE_SOURCES
    assert body["search_query"] == "why are musicians protesting"
    assert isinstance(body["conversation_id"], int)
    # Search got the original query and the requested size.
    assert fake_search.calls == [{"query": "why are musicians protesting", "max_results": 3}]
    # Exactly one LLM call (the answer): no rewrite on the first turn.
    assert len(fake_llm.calls) == 1
    prompt = fake_llm.calls[0][-1].content
    assert "[1] Title One (https://example.com/1)" in prompt
    assert "[2] Title Two (https://example.com/2)" in prompt
    assert "Question: why are musicians protesting" in prompt


def test_conversation_is_persisted_without_rerunning_anything():
    reset()
    convs = client.get("/conversations").json()["conversations"]
    assert len(convs) == 1
    msgs = client.get(f"/conversations/{convs[0]['id']}").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["sources"][0]["url"] == "https://example.com/1"
    assert fake_search.calls == [] and fake_llm.calls == []  # pure DB read


def test_follow_up_is_rewritten_and_appended():
    reset()
    conv_id = client.get("/conversations").json()["conversations"][0]["id"]
    history = [
        {"role": "user", "content": "why are musicians protesting"},
        {"role": "assistant", "content": "Answer text [1][2]."},
    ]
    r = client.post(
        "/ask",
        json={"query": "which one started first?", "history": history, "conversation_id": conv_id},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["search_query"] == "rewritten standalone query"
    assert fake_search.calls[0]["query"] == "rewritten standalone query"
    assert len(fake_llm.calls) == 2  # rewrite + answer
    assert body["conversation_id"] == conv_id
    assert len(client.get("/conversations").json()["conversations"]) == 1  # no duplicate
    msgs = client.get(f"/conversations/{conv_id}").json()["messages"]
    assert len(msgs) == 4


def test_no_results_does_not_call_the_llm():
    reset()
    fake_search.results = []
    r = client.post("/ask", json={"query": "zzz no results"})
    assert r.status_code == 200
    body = r.json()
    assert body["sources"] == []
    assert "could not find" in body["answer"]
    assert fake_llm.calls == []


def test_search_failure_returns_502_and_leaves_no_empty_conversation():
    reset()
    before = len(client.get("/conversations").json()["conversations"])
    fake_search.error = WebSearchError("boom")
    r = client.post("/ask", json={"query": "will fail"})
    assert r.status_code == 502
    assert "boom" in r.json()["detail"]
    assert len(client.get("/conversations").json()["conversations"]) == before


def test_empty_llm_answer_returns_502_and_stores_nothing():
    reset()
    before = len(client.get("/conversations").json()["conversations"])
    fake_llm.answer = ""
    r = client.post("/ask", json={"query": "empty answer case"})
    assert r.status_code == 502
    assert len(client.get("/conversations").json()["conversations"]) == before


def test_top_k_is_clamped():
    reset()
    client.post("/ask", json={"query": "clamp high", "top_k": 100})
    client.post("/ask", json={"query": "clamp low", "top_k": 0})
    assert [c["max_results"] for c in fake_search.calls] == [main.MAX_RESULTS_LIMIT, 1]


def test_search_endpoint_returns_results_without_llm():
    reset()
    r = client.post("/search", json={"query": "plain search", "top_k": 2})
    assert r.status_code == 200
    assert r.json()["results"] == FAKE_SOURCES
    assert fake_llm.calls == []


def test_cors_allows_known_origin_and_blocks_unknown():
    headers = {"Access-Control-Request-Method": "POST"}
    ok = client.options("/ask", headers={**headers, "Origin": "https://ai-powered-search-engine-rosy.vercel.app"})
    assert ok.status_code == 200
    assert ok.headers.get("access-control-allow-origin") == "https://ai-powered-search-engine-rosy.vercel.app"
    bad = client.options("/ask", headers={**headers, "Origin": "https://evil.example"})
    assert bad.status_code == 400


def test_adapter_normalizes_tavily_response_shape():
    import search

    class FakeTavily:
        def __init__(self):
            self.kwargs = None

        def search(self, **kwargs):
            self.kwargs = kwargs
            # Same field layout as a real Tavily "basic" response.
            return {
                "query": kwargs["query"],
                "results": [
                    {"url": "https://a.example/x", "title": "A", "content": "text a", "score": 0.7, "raw_content": None, "id": "1"},
                    {"url": "https://b.example/y", "title": None, "content": "text b", "score": 0.5, "raw_content": None, "id": "2"},
                    {"url": None, "title": "no url", "content": "dropped", "score": 0.4, "raw_content": None, "id": "3"},
                    {"url": "https://c.example/z", "title": "C", "content": "   ", "score": 0.3, "raw_content": None, "id": "4"},
                ],
            }

    fake = FakeTavily()
    original = search._client
    search._client = fake
    try:
        out = search.web_search("q", max_results=4)
    finally:
        search._client = original

    assert fake.kwargs["search_depth"] == "basic" and fake.kwargs["max_results"] == 4
    assert out == [
        {"title": "A", "url": "https://a.example/x", "snippet": "text a", "score": 0.7},
        {"title": "https://b.example/y", "url": "https://b.example/y", "snippet": "text b", "score": 0.5},
    ]


def test_adapter_wraps_provider_errors():
    import search

    class BrokenTavily:
        def search(self, **kwargs):
            raise RuntimeError("quota exceeded")

    original = search._client
    try:
        search._client = BrokenTavily()
        try:
            search.web_search("q")
            assert False, "expected WebSearchError"
        except WebSearchError as e:
            assert "RuntimeError" in str(e) and "quota exceeded" in str(e)

        search._client = None
        try:
            search.web_search("q")
            assert False, "expected WebSearchError"
        except WebSearchError as e:
            assert "TAVILY_API_KEY" in str(e)
    finally:
        search._client = original


def _ask_through_real_adapter(tavily_results, query="adapter filter case"):
    """POST /ask with the REAL search.web_search (and its MIN_SCORE filter)
    sitting between a fake Tavily client and main. Only Tavily is faked."""
    import search

    class FakeTavily:
        def search(self, **kwargs):
            return {"results": tavily_results}

    saved = (main.web_search, search._client, search.MIN_SCORE)
    main.web_search = search.web_search
    search._client = FakeTavily()
    search.MIN_SCORE = 0.20  # pin it so a MIN_SCORE value in .env cannot change the test
    try:
        return client.post("/ask", json={"query": query})
    finally:
        main.web_search, search._client, search.MIN_SCORE = saved


def test_all_results_below_min_score_skips_the_llm():
    reset()
    # The real junk scores seen when "first" matched robotics/security sites.
    junk = [
        {"url": f"https://junk.example/{i}", "title": f"Junk {i}", "content": "unrelated text", "score": s}
        for i, s in enumerate([0.022, 0.020, 0.016, 0.013, 0.013])
    ]
    r = _ask_through_real_adapter(junk)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sources"] == []
    assert "could not find" in body["answer"]
    assert fake_llm.calls == []  # nothing relevant: the LLM must not be asked


def test_mixed_scores_keep_only_relevant_sources_numbered_from_one():
    reset()
    mixed = [
        {"url": "https://good.example/a", "title": "Good A", "content": "relevant text a", "score": 0.76},
        {"url": "https://junk.example/b", "title": "Junk B", "content": "unrelated text b", "score": 0.022},
        # The real case that slipped past the old 0.05 threshold: an Instagram page of captions.
        {"url": "https://junk.example/e", "title": "Junk E", "content": "unrelated text e", "score": 0.0589},
        {"url": "https://good.example/c", "title": "Good C", "content": "relevant text c", "score": 0.48},
        # No score: cannot be judged, so it is kept rather than silently dropped.
        {"url": "https://noscore.example/d", "title": "No Score D", "content": "text d", "score": None},
    ]
    r = _ask_through_real_adapter(mixed)
    assert r.status_code == 200, r.text
    body = r.json()
    assert [s["title"] for s in body["sources"]] == ["Good A", "Good C", "No Score D"]
    prompt = fake_llm.calls[0][-1].content
    # [n] is renumbered AFTER filtering, so there is no gap where Junk B was.
    assert "[1] Good A (https://good.example/a)" in prompt
    assert "[2] Good C (https://good.example/c)" in prompt
    assert "[3] No Score D (https://noscore.example/d)" in prompt
    assert "Junk B" not in prompt and "unrelated text b" not in prompt
    assert "Junk E" not in prompt and "unrelated text e" not in prompt


def test_rewrite_prompt_carries_history_and_falls_back_when_empty():
    reset()
    history = [
        main.ChatTurn(role="user", content="why are musicians protesting"),
        main.ChatTurn(
            role="assistant",
            content="Bands pulled their music from Spotify over the CEO's defense-industry investments [5].",
        ),
    ]
    follow_up = "tell me more about the Spotify protest"

    assert main.rewrite_query_with_history(follow_up, history) == "rewritten standalone query"
    prompt = fake_llm.calls[0]  # the rewrite call receives one plain string
    assert isinstance(prompt, str)
    # The specific detail the rewriter must preserve is actually in front of it.
    assert "the CEO's defense-industry investments" in prompt
    assert f"Follow-up question: {follow_up}" in prompt

    # An empty rewrite (reasoning model out of tokens) must not become an empty search.
    fake_llm.rewrite = "   "
    try:
        assert main.rewrite_query_with_history(follow_up, history) == follow_up
    finally:
        fake_llm.rewrite = "rewritten standalone query"

def test_strip_reference_lines_removes_wikipedia_footnotes():
    import search

    footnotes_only = (
        '15. ↑ Corlett, Eva (May 9, 2024). "Macklemore performs pro-Palestine song". The Guardian. Retrieved May 9, 2024.\n'
        '16. ↑ Lau, Evelyn (May 11, 2024). "Hind\'s Hall is now on streaming". The National. Retrieved May 12, 2024. [...] '
        '8. 1 2 3 4 5 "US rapper Macklemore releases track". The New Arab. May 7, 2024. Archived from the original on May 8, 2024. Retrieved May 8, 2024.'
    )
    assert search.strip_reference_lines(footnotes_only) == ""

    mixed = (
        '"Hind\'s Hall" is a protest song. Released as a single on May 6, 2024, it expresses support. [...] '
        '13. ↑ Blistein, Jon (May 7, 2024). "Macklemore Drops". Rolling Stone. Retrieved May 9, 2024.'
    )
    out = search.strip_reference_lines(mixed)
    assert "Released as a single on May 6, 2024" in out
    assert "May 7, 2024" not in out and "Blistein" not in out


def test_strip_reference_lines_keeps_normal_text():
    import search

    text = "Plain sentence. [...] Another one."
    assert search.strip_reference_lines(text) == text
    recipe = "1. 2 eggs\n2. 1 cup flour"  # numbered lines without footnote markers stay
    assert search.strip_reference_lines(recipe) == recipe


def test_footnote_only_source_is_dropped_by_adapter():
    reset()
    results = [
        {
            "url": "https://en.wikipedia.org/wiki/X",
            "title": "X",
            "content": '13. ↑ Blistein, Jon (May 7, 2024). "Title". Rolling Stone. Retrieved May 9, 2024.',
            "score": 0.8,
        },
        {"url": "https://good.example/a", "title": "Good", "content": "Real text.", "score": 0.7},
    ]
    r = _ask_through_real_adapter(results)
    assert r.status_code == 200, r.text
    assert [s["title"] for s in r.json()["sources"]] == ["Good"]

# ---------------------------------------------------------------- rate limit

def _make_limiter(ip_per_minute=5, ip_per_day=10, global_per_day=30, client=None):
    """A RateLimiter backed by an isolated in-memory fake Redis that really
    runs the Lua script (needs: pip install "fakeredis[lua]")."""
    import fakeredis
    import rate_limit

    client = client or fakeredis.FakeRedis(server=fakeredis.FakeServer())
    return rate_limit.RateLimiter(client, ip_per_minute, ip_per_day, global_per_day), client


def test_rate_limit_per_ip_minute_blocks_extra_requests():
    limiter, _ = _make_limiter(ip_per_minute=5, ip_per_day=0, global_per_day=0)
    assert [limiter.check("1.1.1.1")[0] for _ in range(5)] == [True] * 5
    allowed, limit_hit, retry_after = limiter.check("1.1.1.1")
    assert (allowed, limit_hit) == (False, "ip_minute")
    assert 1 <= retry_after <= 60
    assert limiter.check("2.2.2.2")[0] is True  # another IP is not affected


def test_rate_limit_global_cap_blocks_every_ip_and_rejections_cost_nothing():
    limiter, client = _make_limiter(ip_per_minute=0, ip_per_day=2, global_per_day=3)
    assert limiter.check("1.1.1.1")[0] and limiter.check("1.1.1.1")[0]  # IP A hits its daily 2
    assert limiter.check("1.1.1.1")[:2] == (False, "ip_day")
    assert limiter.check("2.2.2.2")[0] is True  # global is now 3 of 3
    allowed, limit_hit, retry_after = limiter.check("3.3.3.3")
    assert (allowed, limit_hit) == (False, "global_day")
    assert 1 <= retry_after <= 86400
    # A rejected request must not consume quota on the other counters.
    assert client.get("rl:ip_day:3.3.3.3") is None
    assert int(client.get("rl:ip_day:1.1.1.1")) == 2


def test_rate_limited_ask_returns_429_and_never_reaches_tavily():
    import rate_limit

    reset()
    original = rate_limit.limiter
    rate_limit.limiter, _ = _make_limiter(ip_per_minute=1, ip_per_day=0, global_per_day=0)
    try:
        headers = {"x-forwarded-for": "9.9.9.9"}
        first = client.post("/ask", json={"query": "first call"}, headers=headers)
        second = client.post("/ask", json={"query": "second call"}, headers=headers)
        search_second = client.post("/search", json={"query": "third call"}, headers=headers)
    finally:
        rate_limit.limiter = original
    assert first.status_code == 200, first.text
    assert second.status_code == 429
    detail = second.json()["detail"]
    assert detail["code"] == "rate_limited" and detail["limit"] == "ip_minute"
    assert second.headers["retry-after"] == str(detail["retry_after"])
    assert search_second.status_code == 429  # /search shares the same budget
    # Only the first request spent a search credit.
    assert [c["query"] for c in fake_search.calls] == ["first call"]


def test_client_ip_is_read_from_the_right_of_x_forwarded_for():
    from starlette.requests import Request
    import rate_limit

    def make(header, peer="10.0.0.1"):
        headers = [(b"x-forwarded-for", header.encode())] if header else []
        return Request({"type": "http", "headers": headers, "client": (peer, 1234)})

    # One trusted proxy appended the real client; a forged value sits on the left.
    assert rate_limit.get_client_ip(make("6.6.6.6, 7.7.7.7"), 1) == "7.7.7.7"
    assert rate_limit.get_client_ip(make("6.6.6.6, 7.7.7.7"), 2) == "6.6.6.6"
    assert rate_limit.get_client_ip(make(""), 1) == "10.0.0.1"          # no header
    assert rate_limit.get_client_ip(make("7.7.7.7"), 2) == "10.0.0.1"   # fewer entries than hops
    assert rate_limit.get_client_ip(make("7.7.7.7"), 0) == "10.0.0.1"   # header ignored


def test_redis_down_fails_closed_by_default_and_open_when_configured():
    import redis
    import rate_limit

    class BrokenRedis:
        def register_script(self, script):
            def run(keys, args):
                raise redis.ConnectionError("upstash unreachable")
            return run

    reset()
    original_limiter, original_open = rate_limit.limiter, rate_limit.RATE_LIMIT_FAIL_OPEN
    rate_limit.limiter = rate_limit.RateLimiter(BrokenRedis(), 5, 10, 30)
    try:
        rate_limit.RATE_LIMIT_FAIL_OPEN = False
        closed = client.post("/ask", json={"query": "redis down closed"})
        assert closed.status_code == 503
        assert closed.json()["detail"]["code"] == "rate_limiter_unavailable"
        assert fake_search.calls == []  # credits protected

        rate_limit.RATE_LIMIT_FAIL_OPEN = True
        opened = client.post("/ask", json={"query": "redis down open"})
        assert opened.status_code == 200, opened.text
    finally:
        rate_limit.limiter, rate_limit.RATE_LIMIT_FAIL_OPEN = original_limiter, original_open

if __name__ == "__main__":
    # Explicit order: the persistence and follow-up tests depend on earlier state.
    order = [
        "test_health",
        "test_first_turn_numbers_sources_and_skips_rewrite",
        "test_conversation_is_persisted_without_rerunning_anything",
        "test_follow_up_is_rewritten_and_appended",
        "test_no_results_does_not_call_the_llm",
        "test_search_failure_returns_502_and_leaves_no_empty_conversation",
        "test_empty_llm_answer_returns_502_and_stores_nothing",
        "test_top_k_is_clamped",
        "test_search_endpoint_returns_results_without_llm",
        "test_cors_allows_known_origin_and_blocks_unknown",
        "test_adapter_normalizes_tavily_response_shape",
        "test_adapter_wraps_provider_errors",
        "test_all_results_below_min_score_skips_the_llm",
        "test_mixed_scores_keep_only_relevant_sources_numbered_from_one",
        "test_rewrite_prompt_carries_history_and_falls_back_when_empty",
        "test_strip_reference_lines_removes_wikipedia_footnotes",
        "test_strip_reference_lines_keeps_normal_text",
        "test_footnote_only_source_is_dropped_by_adapter",
        "test_rate_limit_per_ip_minute_blocks_extra_requests",
        "test_rate_limit_global_cap_blocks_every_ip_and_rejections_cost_nothing",
        "test_rate_limited_ask_returns_429_and_never_reaches_tavily",
        "test_client_ip_is_read_from_the_right_of_x_forwarded_for",
        "test_redis_down_fails_closed_by_default_and_open_when_configured",
    ]
    failed = 0
    for name in order:
        try:
            globals()[name]()
            print(f"PASS {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    os.unlink(_tmp.name)
    sys.exit(1 if failed else 0)