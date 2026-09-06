"""
debug_rerank.py — Xem toàn bộ candidate + điểm rerank thật, TRƯỚC khi bị lọc/cắt.
Dùng để chẩn đoán: 1 article bị loại vì điểm thấp thật (rerank đúng), hay vì
hybrid search ngay từ đầu không đưa nó vào top 15 candidate (vấn đề retrieval,
không phải rerank).
Chạy: python debug_rerank.py
"""

from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from fastembed.rerank.cross_encoder import TextCrossEncoder

from config import QDRANT_URL, EMBEDDING_MODEL, COLLECTION_NAME

SPARSE_MODEL_NAME = "Qdrant/bm25"
RERANK_MODEL_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"
CANDIDATE_POOL = 15

QUERY = "why are musicians protesting"
WATCH_ARTICLE_ID = 2076  # article về file-sharing, muốn biết nó rơi vào đâu


def main():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL_NAME)
    client = QdrantClient(url=QDRANT_URL)
    reranker = TextCrossEncoder(model_name=RERANK_MODEL_NAME)

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )

    candidates = vectorstore.similarity_search(QUERY, k=CANDIDATE_POOL)
    print(f"Hybrid search trả về {len(candidates)} candidate cho query: \"{QUERY}\"\n")

    article_ids_in_pool = [doc.metadata.get("article_id") for doc in candidates]
    if WATCH_ARTICLE_ID in article_ids_in_pool:
        print(f"[OK] article_id={WATCH_ARTICLE_ID} CÓ nằm trong pool {CANDIDATE_POOL} candidate ban đầu.\n")
    else:
        print(f"[VẤN ĐỀ] article_id={WATCH_ARTICLE_ID} KHÔNG nằm trong pool {CANDIDATE_POOL} candidate ban đầu.")
        print("  → Đây là vấn đề ở RETRIEVAL (hybrid search), không phải rerank — cross-encoder")
        print(f"    không có cơ hội chấm điểm cho article {WATCH_ARTICLE_ID} vì nó chưa từng lọt vào danh sách.\n")

    candidate_texts = [doc.page_content for doc in candidates]
    scores = list(reranker.rerank(QUERY, candidate_texts))

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    print(f"{'Hạng':<6}{'article_id':<12}{'category':<15}{'rerank_score':<15}")
    for rank, (doc, score) in enumerate(scored, start=1):
        aid = doc.metadata.get("article_id")
        cat = doc.metadata.get("category")
        marker = "  <-- ĐANG THEO DÕI" if aid == WATCH_ARTICLE_ID else ""
        print(f"{rank:<6}{aid:<12}{cat:<15}{score:<15.3f}{marker}")


if __name__ == "__main__":
    main()
