from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def filter_sentences(query, docs, top_k=5):

    sentences = []

    for doc in docs:
        text = doc.page_content if hasattr(doc, "page_content") else str(doc)
        for s in text.split("."):
            if len(s.strip()) > 20:
                sentences.append(s.strip())

    pairs = [(query, s) for s in sentences]

    scores = model.predict(pairs)

    ranked = list(zip(sentences, scores))

    ranked.sort(key=lambda x: x[1], reverse=True)

    best_sentences = [s for s, score in ranked[:top_k]]

    return "\n".join(best_sentences)