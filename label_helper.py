"""
label_helper.py — Công cụ hỗ trợ tự gán nhãn cho golden set.
Chạy: python label_helper.py

Với mỗi câu hỏi candidate, chạy hybrid search, in ra đầy đủ text + article_id
của top 10 kết quả để BẠN tự đọc và xác nhận cái nào thực sự đúng — không có
cách nào khác để biết "đáp án đúng" ngoài việc con người tự đọc và quyết định,
đây chính là cách các bộ dữ liệu evaluation thật (MS MARCO, BEIR...) được xây.
"""

from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient

from config import QDRANT_URL, EMBEDDING_MODEL, COLLECTION_NAME

SPARSE_MODEL_NAME = "Qdrant/bm25"

# Câu hỏi candidate — chia theo 3 nhóm để đo sau này biết hybrid mạnh/yếu ở đâu.
# Thêm/sửa tuỳ ý theo dataset thật của bạn.
CANDIDATES = {
    "ten_rieng": [
        "u2",
        "napster",
        "radiohead",
    ],
    "paraphrase": [
        "movie release delayed",
        "illegal downloading lawsuits",
        "government funding for the arts",
    ],
    "cau_hoi_thuong": [
        "latest technology gadgets",
        "sports championship results",
        "economic policy changes",
    ],
}


def build_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL_NAME)
    client = QdrantClient(url=QDRANT_URL)
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )


if __name__ == "__main__":
    vectorstore = build_vectorstore()

    for group_name, queries in CANDIDATES.items():
        print(f"\n{'=' * 60}")
        print(f"NHÓM: {group_name}")
        print(f"{'=' * 60}")

        for query in queries:
            print(f"\n--- Query: \"{query}\" ---")
            results = vectorstore.similarity_search(query, k=10)

            seen_articles = {}
            for doc in results:
                aid = doc.metadata.get("article_id")
                if aid not in seen_articles:
                    seen_articles[aid] = doc.page_content

            for aid, text in seen_articles.items():
                print(f"  article_id={aid}: {text[:200]}...")

            print("  >>> Đọc các đoạn trên, ghi lại article_id nào THỰC SỰ trả lời đúng query này")
