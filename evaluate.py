from collections import defaultdict

from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient

from config import QDRANT_URL, EMBEDDING_MODEL, COLLECTION_NAME

SPARSE_MODEL_NAME = "Qdrant/bm25"
TOP_K = 5

GOLDEN_SET = [
    {"query": "kapranos", "expected_article_ids": {233, 386}, "category": "proper_noun"},
    {"query": "u2", "expected_article_ids": {371, 259, 1, 385, 226, 76}, "category": "proper_noun"},
    {"query": "napster", "expected_article_ids": {1844, 1891}, "category": "proper_noun"},

    {"query": "why are musicians protesting", "expected_article_ids": {0, 2076}, "category": "paraphrase"},
    {"query": "movie release delayed", "expected_article_ids": {305, 38, 365}, "category": "paraphrase"},
    {"query": "illegal downloading lawsuits", "expected_article_ids": {1977, 2166, 2220, 1989}, "category": "paraphrase"},
    {"query": "government funding for the arts", "expected_article_ids": {386, 1556}, "category": "paraphrase"},

    {"query": "latest technology gadgets", "expected_article_ids": {2199, 1920, 2066, 1940, 2169, 2131, 1872}, "category": "general"},
    {"query": "who won the australian open tennis title", "expected_article_ids": {1002}, "category": "general"},
    {"query": "eu stability pact deficit rules changed", "expected_article_ids": {1533, 854}, "category": "general"},
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
        return QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
            retrieval_mode=RetrievalMode.DENSE,
            vector_name="dense",
        )


def score_one_query(vectorstore, case):
    results = vectorstore.similarity_search(case["query"], k=TOP_K)
    retrieved_ids = [doc.metadata.get("article_id") for doc in results]

    hit = any(rid in case["expected_article_ids"] for rid in retrieved_ids)
    correct_count = sum(1 for rid in retrieved_ids if rid in case["expected_article_ids"])
    precision = correct_count / len(retrieved_ids) if retrieved_ids else 0
    rank = next(
        (i + 1 for i, rid in enumerate(retrieved_ids) if rid in case["expected_article_ids"]),
        None,
    )
    reciprocal_rank = 1 / rank if rank else 0

    return {
        "query": case["query"],
        "retrieved_ids": retrieved_ids,
        "hit": hit,
        "precision": precision,
        "reciprocal_rank": reciprocal_rank,
        "rank": rank,
    }


def print_group_summary(group_name, scored_cases):
    n = len(scored_cases)
    recall = sum(1 for c in scored_cases if c["hit"]) / n
    mrr = sum(c["reciprocal_rank"] for c in scored_cases) / n
    precision = sum(c["precision"] for c in scored_cases) / n
    print(f"    [{group_name}] n={n} | Recall@{TOP_K}={recall:.2f} | MRR={mrr:.3f} | Precision@{TOP_K}={precision:.2f}")


def evaluate(vectorstore: QdrantVectorStore, label: str):
    print(f"\n=== {label} ===")

    by_category = defaultdict(list)

    for case in GOLDEN_SET:
        if not case["expected_article_ids"]:
            print(f"  [bỏ qua] \"{case['query']}\" — chưa có expected_article_ids")
            continue

        scored = score_one_query(vectorstore, case)
        by_category[case["category"]].append(scored)

        status = f"hit @ rank {scored['rank']}" if scored["rank"] else "MISS"
        print(f"  ({case['category']}) \"{case['query']}\" → {scored['retrieved_ids']} → {status}, precision={scored['precision']:.2f}")

    print()
    all_scored = []
    for category, scored_cases in by_category.items():
        print_group_summary(category, scored_cases)
        all_scored.extend(scored_cases)

    if all_scored:
        print_group_summary("TỔNG (mọi nhóm gộp lại)", all_scored)


if __name__ == "__main__":
    dense_only_store = build_vectorstore(RetrievalMode.DENSE)
    evaluate(dense_only_store, "DENSE-ONLY (chỉ semantic search)")

    hybrid_store = build_vectorstore(RetrievalMode.HYBRID)
    evaluate(hybrid_store, "HYBRID (dense + sparse)")