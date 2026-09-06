"""
evaluate.py — Đo chất lượng retrieval bằng golden set (query đã biết đáp án đúng).
Chạy: python evaluate.py

Đo 2 chỉ số chuẩn ngành Information Retrieval:
- Recall@k: trong top-k kết quả trả về, có bao nhiêu % câu hỏi tìm được ÍT NHẤT 1
  article đúng như kỳ vọng. Đo "có tìm ra được không", không quan tâm thứ hạng.
- MRR (Mean Reciprocal Rank): trung bình của 1/(vị_trí article đúng đầu tiên xuất hiện).
  Đo "tìm ra đúng NHANH tới đâu" — article đúng đứng #1 thì tốt hơn đứng #5, dù cả
  2 đều tính là "recall thành công".
"""

from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient

from config import QDRANT_URL, EMBEDDING_MODEL, COLLECTION_NAME

SPARSE_MODEL_NAME = "Qdrant/bm25"
TOP_K = 5

# Golden set — mỗi query đi kèm 1 tập article_id được coi là "đúng".
# Đây là bộ tự xây dựa trên các case đã tự tay xác minh trong quá trình phát triển —
# không phải nhãn khách quan tuyệt đối, nhưng đủ tin cậy vì đã đọc và xác nhận nội dung thật.
GOLDEN_SET = [
    {
        "query": "kapranos",
        "expected_article_ids": {233, 386},
        "note": "Tên riêng hiếm — case chứng minh giá trị của sparse/hybrid search",
    },
    {
        "query": "why are musicians protesting",
        "expected_article_ids": {0, 2076},
        "note": "Câu hỏi ngữ nghĩa — 2 chủ đề khác nhau (visa, file-sharing)",
    },
    {
        "query": "movie release delayed",
        "expected_article_ids": set(),  # để trống nếu chưa xác minh đáp án cụ thể — script sẽ bỏ qua khi tính điểm
        "note": "Case paraphrase — cần tự xác minh article_id đúng trước khi đưa vào chấm điểm",
    },
]


def build_vectorstore(retrieval_mode: RetrievalMode) -> QdrantVectorStore:
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    client = QdrantClient(url=QDRANT_URL)

    if retrieval_mode == RetrievalMode.HYBRID:
        sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL_NAME)
        return QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
    else:
        # DENSE mode reuses the same "dense" named vector, just skips the sparse side —
        # this lets us compare hybrid vs dense-only on the EXACT same stored data,
        # instead of guessing at the difference.
        return QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
            retrieval_mode=RetrievalMode.DENSE,
            vector_name="dense",
        )


def evaluate(vectorstore: QdrantVectorStore, label: str):
    print(f"\n=== {label} ===")

    recall_hits = 0
    reciprocal_ranks = []
    scored_queries = 0

    for case in GOLDEN_SET:
        if not case["expected_article_ids"]:
            print(f"  [bỏ qua] \"{case['query']}\" — chưa có expected_article_ids")
            continue

        scored_queries += 1
        results = vectorstore.similarity_search(case["query"], k=TOP_K)
        retrieved_ids = [doc.metadata.get("article_id") for doc in results]

        # Recall@k: có ít nhất 1 ID đúng nằm trong top-k không?
        hit = any(rid in case["expected_article_ids"] for rid in retrieved_ids)
        if hit:
            recall_hits += 1

        # MRR: tìm vị trí (1-indexed) của kết quả đúng ĐẦU TIÊN
        rank = next(
            (i + 1 for i, rid in enumerate(retrieved_ids) if rid in case["expected_article_ids"]),
            None,
        )
        reciprocal_ranks.append(1 / rank if rank else 0)

        status = f"hit @ rank {rank}" if rank else "MISS"
        print(f"  \"{case['query']}\" → article_ids trả về: {retrieved_ids} → {status}")

    if scored_queries == 0:
        print("  Không có query nào đủ dữ liệu để chấm điểm.")
        return

    recall_at_k = recall_hits / scored_queries
    mrr = sum(reciprocal_ranks) / scored_queries

    print(f"\n  Recall@{TOP_K}: {recall_at_k:.2f} ({recall_hits}/{scored_queries} query tìm đúng)")
    print(f"  MRR: {mrr:.3f}")


if __name__ == "__main__":
    dense_only_store = build_vectorstore(RetrievalMode.DENSE)
    evaluate(dense_only_store, "DENSE-ONLY (chỉ semantic search)")

    hybrid_store = build_vectorstore(RetrievalMode.HYBRID)
    evaluate(hybrid_store, "HYBRID (dense + sparse)")
