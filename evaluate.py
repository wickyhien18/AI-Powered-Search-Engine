from collections import defaultdict

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient

from config import QDRANT_URL, EMBEDDING_MODEL, COLLECTION_NAME, QDRANT_API_KEY

SPARSE_MODEL_NAME = "Qdrant/bm25"
TOP_K = 5

GOLDEN_SET = [
    # ===== BUSINESS =====
    {"query": "qantas profits", "expected_article_ids": {409}, "category": "proper_noun"},
    {"query": "axa sun life bonus cuts", "expected_article_ids": {806}, "category": "proper_noun"},
    {"query": "dollar weakening against the euro", "expected_article_ids": {759}, "category": "paraphrase"},
    {"query": "contract pulled from iraq reconstruction", "expected_article_ids": {822}, "category": "paraphrase"},
    {"query": "venezuela land reform farms", "expected_article_ids": {571}, "category": "general"},
    {"query": "india aerospace outsourcing jobs", "expected_article_ids": {887}, "category": "general"},
 
    # ===== ENTERTAINMENT =====
    {"query": "jennifer lopez", "expected_article_ids": {105}, "category": "proper_noun"},
    {"query": "harry potter prisoner of azkaban dvd", "expected_article_ids": {265}, "category": "proper_noun"},
    {"query": "actors favored to win at the oscars", "expected_article_ids": {392}, "category": "paraphrase"},
    {"query": "film festival standing ovation for true story", "expected_article_ids": {351}, "category": "paraphrase"},
    {"query": "band chart success after brit awards", "expected_article_ids": {14}, "category": "general"},
    {"query": "sitcom actress returns to tv comedy", "expected_article_ids": {108}, "category": "general"},
 
    # ===== POLITICS =====
    {"query": "alastair campbell peter mandelson bbc row", "expected_article_ids": {1638}, "category": "proper_noun"},
    {"query": "gordon brown budget date", "expected_article_ids": {1437}, "category": "proper_noun"},
    {"query": "government closing overseas embassies to cut costs", "expected_article_ids": {1547}, "category": "paraphrase"},
    {"query": "mps question royal family income", "expected_article_ids": {1803}, "category": "paraphrase"},
    {"query": "preventing terrorist attacks nuclear threat", "expected_article_ids": {1788}, "category": "general"},
    {"query": "party election campaign strategy criticized", "expected_article_ids": {1467}, "category": "general"},
 
    # ===== SPORT =====
    {"query": "andre agassi australian open", "expected_article_ids": {1349}, "category": "proper_noun"},
    {"query": "jonny wilkinson injury return", "expected_article_ids": {1083}, "category": "proper_noun"},
    {"query": "sprinters suspended for failing drug tests", "expected_article_ids": {1246}, "category": "paraphrase"},
    {"query": "football coach fined over racism comments", "expected_article_ids": {1109}, "category": "paraphrase"},
    {"query": "bid to host rugby world cup in asia", "expected_article_ids": {1058}, "category": "general"},
    {"query": "tennis player returning to form after injury", "expected_article_ids": {1256}, "category": "general"},
 
    # ===== TECH =====
    {"query": "apple sues over leaked products", "expected_article_ids": {1889}, "category": "proper_noun"},
    {"query": "disney backs sony dvd format", "expected_article_ids": {2111}, "category": "proper_noun"},
    {"query": "computer viruses used to steal money", "expected_article_ids": {1850}, "category": "paraphrase"},
    {"query": "future of home television technology", "expected_article_ids": {1935}, "category": "paraphrase"},
    {"query": "search engines improving results", "expected_article_ids": {1956}, "category": "general"},
    {"query": "digital divide between rich and poor nations", "expected_article_ids": {2089}, "category": "general"},

    # ==== QUERY =====
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
    embeddings = FastEmbedEmbeddings(model=EMBEDDING_MODEL)
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

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
    elif retrieval_mode == RetrievalMode.SPARSE:
        sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL_NAME)
        return QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.SPARSE,
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

    sparse_only_store = build_vectorstore(RetrievalMode.SPARSE)
    evaluate(sparse_only_store, "SPARSE-ONLY (chỉ BM25 search)")

    hybrid_store = build_vectorstore(RetrievalMode.HYBRID)
    evaluate(hybrid_store, "HYBRID (dense + sparse)")