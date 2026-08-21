"""
main.py — FastAPI search API: embed query via Ollama, search Qdrant, return results
Run: uvicorn main:app --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient

from config import QDRANT_URL, EMBEDDING_MODEL, COLLECTION_NAME

app = FastAPI()

# Created once at startup, reused for every request — avoids reconnecting per request
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
client = QdrantClient(url=QDRANT_URL)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search(req: SearchRequest):
    # Step 1: turn the query text into the same kind of 768-number vector
    # that every chunk in Qdrant already has
    query_vector = embeddings.embed_query(req.query)

    # Step 2: ask Qdrant for the closest stored vectors to this one
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=req.top_k,
    ).points

    # Step 3: reshape Qdrant's raw response into something readable
    return {
        "query": req.query,
        "results": [
            {
                "score": point.score,
                "text": point.payload.get("page_content"),
                "article_id": point.payload.get("article_id"),
                "category": point.payload.get("category"),
            }
            for point in results
        ],
    }