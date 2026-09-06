from typing import Literal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_core.messages import HumanMessage, AIMessage
from qdrant_client import QdrantClient
from fastembed.rerank.cross_encoder import TextCrossEncoder

from config import QDRANT_URL, EMBEDDING_MODEL, LLM_MODEL, COLLECTION_NAME, ENABLE_RERANK

SPARSE_MODEL_NAME = "Qdrant/bm25"  # must match the one used in ingest.py
RERANK_MODEL_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"
# Retrieve more candidates than we actually need, so the reranker has real
# material to sort through — reranking a pool of 5 down to 5 does nothing.
RERANK_CANDIDATE_POOL = 15
# ms-marco-MiniLM cross-encoders output raw scores, NOT a 0-1 probability —
# negative scores mean "the model itself thinks this is not relevant".
# 0 is a reasonable cutoff: keep only chunks the cross-encoder actually
# considers relevant, rather than always force-filling exactly top_k slots.
RERANK_MIN_SCORE = 0.0

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL_NAME)
# Cross-encoder: unlike the embedding models above (which encode query and
# chunk SEPARATELY, then compare vectors), a cross-encoder reads the query
# and a candidate chunk TOGETHER in one pass and outputs a single relevance
# score. This is slower per-pair (can't precompute chunk vectors in advance
# like Qdrant does), which is exactly why it only runs on a small shortlist
# AFTER retrieval, never on the full 7583-chunk collection.
#
# Only loaded when ENABLE_RERANK=true — skips downloading/loading this model
# entirely when rerank is off, since it currently isn't reliable on THIS
# dataset's stripped/lowercased chunk text (see config.py comment).
reranker = TextCrossEncoder(model_name=RERANK_MODEL_NAME) if ENABLE_RERANK else None

# num_predict caps how many tokens the model is allowed to generate.
# Without this, llama3 can ramble past what's actually needed, and every
# extra token costs a full forward pass through the model on CPU — this
# is a direct, real time-saver, not a UX trick like streaming.
llm = ChatOllama(model=LLM_MODEL, temperature=0.2, num_predict=256)

client = QdrantClient(url=QDRANT_URL)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
    vector_name="dense",
    sparse_vector_name="sparse",
)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


# One past turn in the conversation — role tells the LLM who said it.
# "user" = the person's earlier question, "assistant" = the model's earlier answer.
class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    query: str
    top_k: int = 5
    history: list[ChatTurn] = []  # empty by default — old clients calling /ask without history still work


def retrieve_chunks(query: str, top_k: int):
    if not ENABLE_RERANK:
        # Default path — the proven hybrid RRF ranking from Giai đoạn 8/11,
        # already validated by evaluate.py. No cross-encoder involved at all.
        results = vectorstore.similarity_search_with_score(query, k=top_k)
        return [
            {
                "score": score,
                "text": doc.page_content,
                "article_id": doc.metadata.get("article_id"),
                "category": doc.metadata.get("category"),
            }
            for doc, score in results
        ]

    # Step 1 — cast a WIDER net than needed (RERANK_CANDIDATE_POOL, not top_k).
    # Hybrid search (RRF) is good at "roughly relevant fast", not perfectly
    # ordered — the reranker's job is to fix the ORDER of these candidates.
    candidates = vectorstore.similarity_search(query, k=RERANK_CANDIDATE_POOL)
    if not candidates:
        return []

    candidate_texts = [doc.page_content for doc in candidates]

    # Step 2 — score each (query, chunk) pair with the cross-encoder.
    # rerank() returns scores in the SAME order as candidate_texts went in.
    rerank_scores = list(reranker.rerank(query, candidate_texts))

    # Step 3 — sort candidates by the NEW cross-encoder score, take only top_k.
    # This can genuinely reorder results versus what hybrid search returned —
    # a chunk hybrid ranked #4 might jump to #1 here if the cross-encoder
    # judges it more directly relevant once it reads query and chunk together.
    scored_candidates = list(zip(candidates, rerank_scores))
    scored_candidates.sort(key=lambda pair: pair[1], reverse=True)

    # Drop anything below the relevance threshold BEFORE cutting to top_k —
    # this is what fixes the "forced 5th irrelevant source" problem: if only
    # 3 candidates actually clear the bar, we return 3, not 5 padded with junk.
    #
    # KNOWN ISSUE (see config.py, ENABLE_RERANK comment): on this dataset's
    # stripped/lowercased text, the cross-encoder's absolute scores are NOT
    # reliably calibrated — an irrelevant chunk has scored HIGHER than a
    # genuinely relevant one in testing. Keep this threshold logic in place
    # for whenever ingest.py is updated to preserve natural-language text,
    # but do not trust it blindly while ENABLE_RERANK is manually turned on.
    relevant_candidates = [pair for pair in scored_candidates if pair[1] >= RERANK_MIN_SCORE]
    top_candidates = relevant_candidates[:top_k]

    return [
        {
            "score": float(score),
            "text": doc.page_content,
            "article_id": doc.metadata.get("article_id"),
            "category": doc.metadata.get("category"),
        }
        for doc, score in top_candidates
    ]


def rewrite_query_with_history(query: str, history: list[ChatTurn]) -> str:
    """
    Turn a context-dependent follow-up ("which category does that belong to?")
    into a standalone question that makes sense on its own ("which category
    does the article about musicians protesting visa costs belong to?").

    This is a SEPARATE, small LLM call that runs BEFORE retrieval — its only
    job is producing better search text, it never sees the Qdrant results and
    its output is never shown to the user directly.
    """
    if not history:
        # First turn in a conversation has nothing to rewrite against — skip
        # the extra LLM call entirely (saves time, and there's nothing to fix).
        return query

    history_text = "\n".join(
        f"{'User' if turn.role == 'user' else 'Assistant'}: {turn.content}"
        for turn in history
    )

    rewrite_prompt = f"""Given this conversation history and a follow-up question, rewrite the \
follow-up question into a standalone question that includes all necessary context \
from the history. If the follow-up question is already standalone, return it unchanged. \
Return ONLY the rewritten question, nothing else — no explanation, no quotes.

Conversation history:
{history_text}

Follow-up question: {query}

Standalone question:"""

    response = llm.invoke(rewrite_prompt)
    return response.content.strip()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search(req: SearchRequest):
    results = retrieve_chunks(req.query, req.top_k)
    return {"query": req.query, "results": results}


@app.post("/ask")
def ask(req: AskRequest):
    # Rewrite BEFORE retrieval — this fixes the exact bug from before:
    # "which category does that article belong to?" becomes something like
    # "which category does the article about musicians protesting visa costs
    # belong to?" — THAT rewritten text is what gets embedded and searched,
    # so retrieval finds the right chunks instead of chunks matching the
    # literal (context-free) words "which category does that article belong to".
    search_query = rewrite_query_with_history(req.query, req.history)

    chunks = retrieve_chunks(search_query, req.top_k)

    context_block = "\n\n".join(
        f"[{i + 1}] (Article #{c['article_id']}, category: {c['category']})\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    current_turn_prompt = f"""Answer the question using ONLY the information in the numbered sources below.
The answer may not appear as one single sentence — combine relevant details from multiple sources if needed.
Only say the sources don't contain the answer if NONE of them are relevant at all.
Do not add outside information. If the question refers back to something from earlier in our
conversation (e.g. "that", "it", "the one you mentioned"), use the conversation history to
understand what is being referred to.

IMPORTANT — citation rule: after EVERY factual claim you make, add the source number in
brackets right after it, like this: "Musicians opposed the lawsuits [2]." Use ONLY the
numbers shown below. If a sentence combines facts from two sources, cite both: "...[1][3]".
Never invent a number that isn't listed below.

Sources:
{context_block}

Question: {req.query}

Answer (with [n] citations after each claim):"""

    # Rebuild the conversation as a real list of typed messages, not one flat string —
    # this is what lets the model tell "who said what" apart, the same way ChatGPT's
    # own API expects a messages list rather than a single blob of text.
    messages = []
    for turn in req.history:
        if turn.role == "user":
            messages.append(HumanMessage(content=turn.content))
        else:
            messages.append(AIMessage(content=turn.content))
    messages.append(HumanMessage(content=current_turn_prompt))

    response = llm.invoke(messages)

    return {
        "query": req.query,
        "answer": response.content,
        "sources": chunks,
    }