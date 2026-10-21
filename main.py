from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel

import db
from config import CORS_ORIGINS, GROQ_API_KEY, LLM_MODEL
from rate_limit import enforce_rate_limit
from search import WebSearchError, web_search

# Upper bound on web results per question, so one request cannot burn
# an unexpectedly large amount of the search API quota.
MAX_RESULTS_LIMIT = 10

app = FastAPI()

db.init_db()  # create the tables if missing; safe to run on every startup

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq-hosted LLM. max_tokens is deliberately high: gpt-oss-20b is a
# REASONING model, so internal "thinking" tokens and the final answer share
# the same budget. Too low a limit exhausts it mid-thought and returns an
# EMPTY answer (a real failure mode hit earlier in this project).
llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0.2, max_tokens=2048)


# One past turn of the conversation. "user" is an earlier question,
# "assistant" is the model's earlier answer.
class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class AskRequest(BaseModel):
    query: str
    top_k: int = 5  # number of web results to retrieve
    history: list[ChatTurn] = []
    # None starts a new conversation (the server creates it and returns its id).
    # A real id appends to that existing conversation.
    conversation_id: int | None = None


def clamp_results(top_k: int) -> int:
    return max(1, min(top_k, MAX_RESULTS_LIMIT))


def rewrite_query_with_history(query: str, history: list[ChatTurn]) -> str:
    """Turn a context-dependent follow-up ("which of them is oldest?") into a
    standalone search query that makes sense on its own.

    This is a separate small LLM call that runs BEFORE the web search. Without
    it, the search engine would receive the literal follow-up text, which has
    no keywords of its own, and return unrelated pages.
    """
    if not history:
        # The first turn has nothing to rewrite against, so skip the extra call.
        return query

    history_text = "\n".join(
        f"{'User' if turn.role == 'user' else 'Assistant'}: {turn.content}"
        for turn in history
    )

    rewrite_prompt = f"""Rewrite the follow-up question into a standalone web search query, using the \
conversation history to work out what the follow-up refers to.

Rules:
- Identify the specific subject the follow-up points at (a person, company, event, product or \
place) and copy its distinguishing details from the history: names, numbers, dates and the \
specific issue involved. If the last answer listed several items, pick the one the follow-up names.
- Do NOT replace those details with generic words. A query that could match several different \
stories is wrong.
- Do NOT add filler such as "details", "context", "overview", "information" or "more about", and \
drop conversational phrasing such as "tell me more".
- Keep it short, like a query typed into a search engine (roughly 4 to 12 words).
- A follow-up often names its subject only in general terms ("the bakers' protest", "that law"). That does NOT make it standalone: if the history describes that subject more specifically (who, what, why), the query MUST carry those specifics. Return the follow-up unchanged ONLY when the history says nothing about its subject, meaning the user has moved to a new topic. Even then, strip conversational phrasing and keep only the topic words.
- Return ONLY the query: no explanation, no quotes.

Example 1 (not part of the real conversation):
History:
User: why are cities banning gas stoves
Assistant: Cities give two reasons. 1) Indoor air pollution from burning gas. 2) Climate targets: \
Berkeley banned gas hookups in new buildings in 2019 [2].
Follow-up question: tell me more about the Berkeley one
Standalone query: Berkeley ban gas hookups new buildings 2019

Example 2 (the follow-up names its subject only generally):
History:
User: why are bakers unhappy
Assistant: Two things are driving it. 1) Flour prices rose 40% after the 2023 drought [1]. 2) Separately, bakers in Lyon quit the national guild in protest of its new flour tax on small shops [3].
Follow-up question: tell me more about the bakers' protest
Standalone query: Lyon bakers quit national guild flour tax small shops

Conversation history:
{history_text}

Follow-up question: {query}

Standalone query:"""

    response = llm.invoke(rewrite_prompt)
    rewritten = response.content.strip()
    # A reasoning model can return an empty string when it runs out of token
    # budget. Searching for "" would fail, so fall back to the original text.
    return rewritten or query


def generate_answer(query: str, history: list[ChatTurn], sources: list[dict]) -> str:
    """Ask the LLM to answer from the numbered web sources, citing them as [n].

    The numbering follows the order of `sources`, so [1] in the answer text is
    always sources[0] in the API response.
    """
    context_block = "\n\n".join(
        f"[{i + 1}] {s['title']} ({s['url']})\n{s['snippet']}"
        for i, s in enumerate(sources)
    )

    current_turn_prompt = f"""Answer the question using ONLY the numbered web sources below.
Each source is a set of excerpts from one web page (excerpts are joined by "[...]").
Excerpts can contain page boilerplate such as navigation text, logos or image captions; ignore it.
Combine details from several sources when needed. If the sources do not contain the answer, \
say so plainly instead of guessing. Do not add outside information.
If the question refers back to the earlier conversation (e.g. "that", "it", "the one you \
mentioned"), use the conversation history to work out exactly which subject is meant and answer \
about that subject only. Ignore sources that are about a different subject.

Accuracy rules:
- Cite a source only for a claim that its own excerpt states. Never cite [2] for something you \
read in [3].
- If two sources give different values for the same fact (a number, a date, a name), report \
both with their own citations instead of picking one.
- If the sources describe different events, campaigns or time periods, keep them separate and \
give the dates. Do not merge them into one movement.
- State a cause, a sequence ("followed", "led to", "because") or a negative ("did not respond") \
only when a source says so explicitly. Otherwise leave it out. Keep the source's hedging words \
("allegedly", "reportedly", "may", "claims") exactly as written; never turn an alleged or reported \
cause into a plain fact.
- Do not copy relative time words such as "last year", "recently", "this month" or "now" from a \
source. Convert them to an absolute year using the year in the source URL or a date in its excerpt \
(for example "in 2025"). If you cannot tell when it happened, leave the time out.
- Do not generalize from examples. Write "all", "every", "none" or "the whole movement" only when a \
source says exactly that. If a source names a few examples, write "several, including X and Y".

Citation rule: after EVERY factual claim, add the source number in brackets, like this: \
"Musicians left the platform [2]." Use ONLY the numbers listed below. If a sentence combines \
facts from two sources, cite both: "...[1][3]." Never invent a number.

Sources:
{context_block}

Question: {query}

Answer (with [n] citations after each claim):"""

    # A real list of typed messages (not one flat string) lets the model tell
    # who said what in the earlier conversation.
    messages = []
    for turn in history:
        if turn.role == "user":
            messages.append(HumanMessage(content=turn.content))
        else:
            messages.append(AIMessage(content=turn.content))
    messages.append(HumanMessage(content=current_turn_prompt))

    response = llm.invoke(messages)
    return response.content


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search", dependencies=[Depends(enforce_rate_limit)])
def search(req: SearchRequest):
    """Web search only, no LLM. Useful for inspecting what the search API returns."""
    try:
        results = web_search(req.query, max_results=clamp_results(req.top_k))
    except WebSearchError as e:
        raise HTTPException(status_code=502, detail=f"Web search failed: {e}")
    return {"query": req.query, "results": results}


@app.post("/ask", dependencies=[Depends(enforce_rate_limit)])
def ask(req: AskRequest):
    # Resolve follow-ups into a standalone search query first.
    search_query = rewrite_query_with_history(req.query, req.history)

    try:
        sources = web_search(search_query, max_results=clamp_results(req.top_k))
    except WebSearchError as e:
        raise HTTPException(status_code=502, detail=f"Web search failed: {e}")

    if sources:
        answer = generate_answer(req.query, req.history, sources)
        if not answer.strip():
            # An empty answer usually means the reasoning model ran out of
            # token budget. Fail loudly instead of storing an empty message.
            raise HTTPException(status_code=502, detail="The model returned an empty answer")
    else:
        # Nothing to ground an answer on, so do not ask the LLM to guess.
        answer = "I could not find any relevant web sources for this question."

    # Persist only after everything succeeded, so a failed request never
    # leaves an empty conversation behind in the sidebar.
    conversation_id = req.conversation_id
    if conversation_id is None:
        title = req.query[:60] + ("..." if len(req.query) > 60 else "")
        conversation_id = db.create_conversation(title)
    db.add_message(conversation_id, "user", req.query)
    db.add_message(conversation_id, "assistant", answer, sources=sources)

    return {
        "query": req.query,
        "search_query": search_query,
        "answer": answer,
        "sources": sources,
        "conversation_id": conversation_id,
    }


@app.get("/conversations")
def get_conversations():
    """Conversation list for the sidebar. Pure database read, no AI involved."""
    return {"conversations": db.list_conversations()}


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: int):
    """All messages of one conversation, loaded straight from the database
    without re-running the search or the LLM."""
    return {"messages": db.get_conversation_messages(conversation_id)}