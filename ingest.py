import glob
import uuid
import pandas as pd

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config import QDRANT_URL, EMBEDDING_MODEL, COLLECTION_NAME, EMBEDDING_DIM


def load_articles() -> list[Document]:
    """ Find .csv file then convert into DataFrame variable and then turn into list of Document"""
    csv_path = glob.glob("./data/*.csv")[0]
    df = pd.read_csv(csv_path)

    documents = []
    for idx, row in df.iterrows():
        doc = Document(
            page_content=row["text"],
            metadata={
                "article_id": int(idx),
                "category": row["category"],
            },
        )
        documents.append(doc)

    print(f"Đã load {len(documents)} bài báo từ {csv_path}")
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split each article into reasonably sized chunks to enable searching that is sufficiently comprehensive, accurate and fast"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, # the max character length any one chunk can hold
        chunk_overlap=50, # how many trailling character from previous chunk get copied into the start of the next one
    )
    chunks = splitter.split_documents(documents)
    print(f"Sau khi chunk: {len(chunks)} chunk (từ {len(documents)} bài báo)")
    return chunks


# Fixed, arbitrary namespace UUID — required by uuid5 to derive deterministic IDs.
# It never changes; it's not a secret, just a seed constant.
ID_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


def make_point_id(chunk: Document, index_in_article: int) -> str:
    """
    Derive a stable ID from the chunk's own content, instead of letting
    Qdrant assign a random one. Same article_id + same chunk position +
    same text -> always the same ID -> re-running ingest.py OVERWRITES
    the existing point instead of inserting a duplicate.
    """
    article_id = chunk.metadata["article_id"]
    unique_string = f"{article_id}_{index_in_article}_{chunk.page_content}"
    return str(uuid.uuid5(ID_NAMESPACE, unique_string))


def assign_point_ids(chunks: list[Document]) -> list[str]:
    """Walk through chunks in order, tracking how many chunks each article_id has
    seen so far, so each chunk gets a distinct, reproducible position index."""
    seen_counts: dict[int, int] = {}
    ids = []
    for chunk in chunks:
        article_id = chunk.metadata["article_id"]
        index_in_article = seen_counts.get(article_id, 0)
        ids.append(make_point_id(chunk, index_in_article))
        seen_counts[article_id] = index_in_article + 1
    return ids


def ensure_collection(client: QdrantClient):
    """Create new connection if not exist - prevent create many connection."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print(f"Đã tạo collection '{COLLECTION_NAME}'")
    else:
        print(f"Collection '{COLLECTION_NAME}' đã tồn tại, dùng lại")


def main():
    documents = load_articles()
    chunks = chunk_documents(documents)
    point_ids = assign_point_ids(chunks)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    client = QdrantClient(url=QDRANT_URL)
    ensure_collection(client)

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    print("Đang embed và lưu vào Qdrant (có thể mất vài phút với CPU)...")
    # Chia batch nhỏ để dễ theo dõi tiến độ và tránh gửi quá nhiều request cùng lúc tới Ollama
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_ids = point_ids[i : i + batch_size]
        vectorstore.add_documents(batch, ids=batch_ids)
        print(f"  Đã xử lý {min(i + batch_size, len(chunks))}/{len(chunks)} chunk")

    print("Ingest hoàn tất.")


if __name__ == "__main__":
    main()