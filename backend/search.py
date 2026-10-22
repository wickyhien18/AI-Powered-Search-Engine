"""
search.py - Thin adapter around the Tavily web search API.

The rest of the app depends only on web_search() and the dict shape it
returns, so switching to another search provider later means editing this
file and nothing else.
"""

import re

from tavily import TavilyClient

from config import MIN_SCORE, TAVILY_API_KEY

EXCERPT_SEPARATOR = "[...]"

# "13. ↑ Author (date). Title. Site. Retrieved ..." is always a reference entry.
_ARROW_FOOTNOTE = re.compile(r"^\s*\d+\.\s*↑")
# Multi-backlink form: '8. 1 2 3 "Title". Site. Archived ... Retrieved ...'
_BACKLINK_FOOTNOTE = re.compile(r"^\s*\d+\.\s+(?:\d+\s+)+")
_CITATION_WORDS = re.compile(r"\b(?:Retrieved|Archived)\b")


def _is_footnote_line(line: str) -> bool:
    if _ARROW_FOOTNOTE.match(line):
        return True
    return bool(_BACKLINK_FOOTNOTE.match(line) and _CITATION_WORDS.search(line))


def strip_reference_lines(snippet: str) -> str:
    """Remove Wikipedia reference-list lines from a Tavily snippet.

    A snippet is several excerpts joined by "[...]". When an excerpt falls in a
    page's References section, the LLM reads article dates as event dates.
    Excerpts left empty are dropped, so a snippet that was only footnotes
    becomes "" and the caller can discard the whole source.
    """
    kept = []
    for excerpt in snippet.split(EXCERPT_SEPARATOR):
        lines = [ln for ln in excerpt.splitlines() if not _is_footnote_line(ln)]
        text = "\n".join(lines).strip()
        if text:
            kept.append(text)
    return f" {EXCERPT_SEPARATOR} ".join(kept)

class WebSearchError(Exception):
    """Raised when the web search provider cannot return results."""


_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Run one basic Tavily search and return normalized sources.

    Each returned dict has: title, url, snippet, score.
    The snippet is Tavily's query-focused extract of the page (several
    excerpts joined by "[...]"), not the full page text.

    Results whose score is below MIN_SCORE are dropped. A result with no score
    is kept, because we cannot judge it and dropping everything would be the
    worse failure if the API ever stopped returning scores.
    """
    if _client is None:
        raise WebSearchError("TAVILY_API_KEY is not set")

    try:
        response = _client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
        )
    except Exception as e:
        raise WebSearchError(f"{type(e).__name__}: {e}") from e

    sources = []
    for result in response.get("results", []):
        url = result.get("url")
        snippet = result.get("content") or ""
        snippet = strip_reference_lines(snippet)
        # A result without a URL cannot be cited, and one without text
        # gives the LLM nothing to work with.
        if not url or not snippet.strip():
            continue
        score = result.get("score")
        # Nearly irrelevant pages (e.g. a keyword match on "first" returning
        # robotics and security sites) must not reach the LLM.
        if score is not None and score < MIN_SCORE:
            continue
        sources.append(
            {
                "title": result.get("title") or url,
                "url": url,
                "snippet": snippet,
                "score": score,
            }
        )
    return sources
