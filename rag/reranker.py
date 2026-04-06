from sentence_transformers import CrossEncoder

print("Loading reranker model...")

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query, docs, top_k=3, return_scores=False):

    if len(docs) == 0:
        return []

    pairs = []

    for doc in docs:
        if hasattr(doc, "page_content"):
            text = doc.page_content
        else:
            text = str(doc)

        pairs.append((query, text))

    scores = reranker.predict(pairs)

    scored_docs = list(zip(docs, scores))

    scored_docs.sort(key=lambda x: x[1], reverse=True)

    top_docs = scored_docs[:top_k]

    if return_scores:
        return top_docs

    return [doc for doc, score in top_docs]