from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
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
llm = ChatOllama(model=LLM_MODEL, temperature=0.2)
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


class AskRequest(BaseModel):
    query: str
    top_k: int = 5


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

    prompt = f"""Answer the question using ONLY the context below. 
    If the context doesn't contain the answer, say so — do not make up information.
 
Context:
{context_block}
 
Question: {req.query}
 
Answer:"""
 
    response = llm.invoke(prompt)
 
    return {
        "query": req.query,
        "answer": response.content,
        "sources": chunks, 
    }
