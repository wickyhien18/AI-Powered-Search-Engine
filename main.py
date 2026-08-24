"""
main.py — FastAPI search API: embed query via Ollama, search Qdrant, return results
Run: uvicorn main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_ollama import OllamaEmbeddings, ChatOllama
from qdrant_client import QdrantClient

from config import QDRANT_URL, EMBEDDING_MODEL, LLM_MODEL, COLLECTION_NAME

app = FastAPI()

# Without this, the browser blocks requests from the Next.js dev server
# (localhost:3000) to this API (localhost:8000) — different ports count
# as different origins as far as browser security is concerned.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Created once at startup, reused for every request — avoids reconnecting per request
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOllama(model=LLM_MODEL, temperature=0.2)
client = QdrantClient(url=QDRANT_URL)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class AskRequest(BaseModel):
    query: str
    top_k: int = 5


def retrieve_chunks(query: str, top_k: int):
    """Shared by /search and /ask — embed the query, fetch nearest chunks from Qdrant."""
    query_vector = embeddings.embed_query(query)
    points = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    ).points

    return [
        {
            "score": point.score,
            "text": point.payload.get("page_content"),
            "article_id": point.payload.get("metadata", {}).get("article_id"),
            "category": point.payload.get("metadata", {}).get("category"),
        }
        for point in points
    ]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search(req: SearchRequest):
    results = retrieve_chunks(req.query, req.top_k)
    return {"query": req.query, "results": results}


@app.post("/ask")
def ask(req: AskRequest):
    # Step 1-2: same retrieval as /search — reused, not duplicated
    chunks = retrieve_chunks(req.query, req.top_k)

    # Step 3: build one prompt string containing the retrieved chunks as "context"
    # This is the entire mechanism of RAG: the LLM never searches anything itself,
    # it only ever sees whatever text we paste into this prompt.
    context_block = "\n\n".join(
        f"[Article #{c['article_id']}, category: {c['category']}]\n{c['text']}"
        for c in chunks
    )

    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say so — do not make up information.

Context:
{context_block}

Question: {req.query}

Answer:"""

    # Step 4: send the prompt to the LLM, get back a generated answer
    response = llm.invoke(prompt)

    return {
        "query": req.query,
        "answer": response.content,
        "sources": chunks,  # so the frontend can show which chunks the answer was based on
    }