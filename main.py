from typing import Literal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_core.messages import HumanMessage, AIMessage
from qdrant_client import QdrantClient

from config import QDRANT_URL, QDRANT_API_KEY, EMBEDDING_MODEL, GROQ_API_KEY, LLM_MODEL, COLLECTION_NAME
import db

SPARSE_MODEL_NAME = "Qdrant/bm25"

app = FastAPI()

db.init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://ai-powered-search-engine-rosy.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL_NAME)

llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0.2, max_tokens=2048)

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)

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
    conversation_id: int | None = None


def retrieve_chunks(query: str, top_k: int):
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
    if not history:
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


@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok"}


@app.post("/search")
def search(req: SearchRequest):
    results = retrieve_chunks(req.query, req.top_k)
    return {"query": req.query, "results": results}


@app.post("/ask")
def ask(req: AskRequest):

    conversation_id = req.conversation_id
    if conversation_id is None:
        title = req.query[:60] + ("..." if len(req.query) > 60 else "")
        conversation_id = db.create_conversation(title)

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

    messages = []
    for turn in req.history:
        if turn.role == "user":
            messages.append(HumanMessage(content=turn.content))
        else:
            messages.append(AIMessage(content=turn.content))
    messages.append(HumanMessage(content=current_turn_prompt))

    response = llm.invoke(messages)

    db.add_message(conversation_id, "user", req.query)
    db.add_message(conversation_id, "assistant", response.content, sources=chunks)

    return {
        "query": req.query,
        "answer": response.content,
        "sources": chunks,
        "conversation_id": conversation_id,
    }


@app.get("/conversations")
def get_conversations():
    return {"conversations": db.list_conversations()}


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: int):
    return {"messages": db.get_conversation_messages(conversation_id)}