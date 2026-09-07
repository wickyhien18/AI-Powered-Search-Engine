from typing import Literal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_core.messages import HumanMessage, AIMessage
from qdrant_client import QdrantClient

from config import QDRANT_URL, EMBEDDING_MODEL, LLM_MODEL, COLLECTION_NAME
import db

SPARSE_MODEL_NAME = "Qdrant/bm25"  # must match the one used in ingest.py

app = FastAPI()

db.init_db()  # tạo bảng nếu chưa có — an toàn gọi mỗi lần server khởi động

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL_NAME)

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
    # None = "start a new conversation" (server creates one and returns its id).
    # A real id = "continue this conversation" — server appends to it instead
    # of creating a duplicate.
    conversation_id: int | None = None


def retrieve_chunks(query: str, top_k: int):
    """Hybrid search: dense (semantic) + sparse (BM25 keyword), fused into
    one ranked list by Qdrant/langchain-qdrant internally (RRF)."""
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
    # Create a new conversation row if this is the first message — the title
    # is just the first ~60 chars of the query, good enough to recognize in
    # a sidebar list without adding a separate "generate a title" LLM call.
    conversation_id = req.conversation_id
    if conversation_id is None:
        title = req.query[:60] + ("..." if len(req.query) > 60 else "")
        conversation_id = db.create_conversation(title)

    # Rewrite BEFORE retrieval — this fixes the exact bug from before:
    # "which category does that article belong to?" becomes something like
    # "which category does the article about musicians protesting visa costs
    # belong to?" — THAT rewritten text is what gets embedded and searched,
    # so retrieval finds the right chunks instead of chunks matching the
    # literal (context-free) words "which category does that article belong to".
    search_query = rewrite_query_with_history(req.query, req.history)

    chunks = retrieve_chunks(search_query, req.top_k)

    # Numbered [1], [2]... in the SAME order as `chunks` — this is what lets
    # sources[0] in the response line up exactly with "[1]" in the answer text,
    # so the frontend (or the person reading it) can trace any claim back to
    # a specific source with zero guesswork. Independent of retrieval method —
    # this works the same whether chunks came from hybrid search alone or,
    # in the past, a reranked list; it's just numbering whatever list it gets.
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

    # Persist AFTER generation succeeds — both the user's question and the
    # model's answer, as two separate rows, in the order they happened.
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
    """Danh sách conversation cho sidebar — endpoint THUẦN ĐỌC, không chạm gì tới AI."""
    return {"conversations": db.list_conversations()}


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: int):
    """Toàn bộ tin nhắn của 1 conversation — LOAD THẲNG từ database,
    không chạy lại retrieval/LLM gì cả. Đây chính là cơ chế 'chọn lại
    conversation cũ không phải chạy lại model' mà bạn muốn."""
    return {"messages": db.get_conversation_messages(conversation_id)}