from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from langchain_core.documents import Document

print("Loading vector database...")

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

db = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding
)

# ------------------------
# Build BM25 corpus
# ------------------------

docs = db.get(limit = 2000)["documents"]

tokenized_corpus = [doc.split() for doc in docs if doc]

if len(tokenized_corpus) == 0:
    print("⚠️ WARNING: No documents found in vector DB")

bm25 = BM25Okapi(tokenized_corpus)

print("BM25 corpus size:", len(tokenized_corpus))


# ------------------------
# Hybrid Search
# ------------------------

def hybrid_search(query, category=None, k=10):

    if category:
        vector_docs = db.similarity_search(
            query,
            k=k,
            filter={"category": category}
        )
    else:
        vector_docs = db.similarity_search(query, k=k)

    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_top_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k]

    bm25_docs = [
        Document(page_content=docs[i]) for i in bm25_top_indices
    ]
    combined = vector_docs + bm25_docs

    unique = {}

    for doc in combined:
        unique[doc.page_content] = doc

    return list(unique.values())