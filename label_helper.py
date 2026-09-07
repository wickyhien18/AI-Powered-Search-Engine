from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient

from config import QDRANT_URL, EMBEDDING_MODEL, COLLECTION_NAME

SPARSE_MODEL_NAME = "Qdrant/bm25"

CANDIDATES = {
    "unique_name": [
        "u2",
        "napster",
        "radiohead",
    ],
    "paraphrase": [
        "movie release delayed",
        "illegal downloading lawsuits",
        "government funding for the arts",
    ],
    "normal question": [
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
        print(f"GROUP: {group_name}")
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

            print(f"{">>>" * 60}")
