from fastapi import FastAPI
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_core.messages import HumanMessage, AIMessage
from qdrant_client import QdrantClient

from config import QDRANT_URL, EMBEDDING_MODEL, LLM_MODEL, COLLECTION_NAME

SPARSE_MODEL_NAME = "Qdrant/bm25"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Created once at startup, reused for every request — avoids reconnecting per request
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL_NAME)
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

class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    query: str
    top_k: int = 5
    history: list[ChatTurn] = []


def retrieve_chunks(query: str, top_k: int):
    """Shared by /search and /ask — embed the query, fetch nearest chunks from Qdrant."""
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
    search_query = rewrite_query_with_history(req.query, req.history)
 
    chunks = retrieve_chunks(search_query, req.top_k)
 
    context_block = "\n\n".join(
        f"[Article #{c['article_id']}, category: {c['category']}]\n{c['text']}"
        for c in chunks
    )
 
    current_turn_prompt = f"""Answer the question using ONLY the information in the context below.
The answer may not appear as one single sentence — combine relevant details from multiple sections if needed.
Only say the context doesn't contain the answer if NONE of the sections are relevant at all.
Do not add outside information. If the question refers back to something from earlier in our
conversation (e.g. "that", "it", "the one you mentioned"), use the conversation history to
understand what is being referred to.
 
Context:
{context_block}
 
Question: {req.query}
 
Answer:"""
 
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

