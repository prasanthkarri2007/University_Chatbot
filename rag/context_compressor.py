from sentence_transformers import CrossEncoder
from rag.hybrid_retriever import hybrid_search


# -----------------------
# Load Cross Encoder Model
# -----------------------

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# -----------------------
# Compression Retriever
# -----------------------

class CompressionRetriever:

    def invoke(self, query, k=10):

        docs = hybrid_search(query, k=20)

        pairs = [(query, doc.page_content) for doc in docs]

        scores = model.predict(pairs)

        scored_docs = list(zip(docs, scores))

        scored_docs.sort(key=lambda x: x[1], reverse=True)

        top_docs = [doc for doc, score in scored_docs[:k]]

        return top_docs


compression_retriever = CompressionRetriever()