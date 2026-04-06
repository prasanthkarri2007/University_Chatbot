from sentence_transformers import SentenceTransformer, util
import numpy as np
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt")

from nltk.tokenize import sent_tokenize


model = SentenceTransformer("all-MiniLM-L6-v2")


def semantic_chunk(text, similarity_threshold=0.65, max_chunk_size=500):
    """
    Split text into semantic chunks based on sentence similarity.
    """

    sentences = sent_tokenize(text)

    if len(sentences) == 0:
        return []

    embeddings = model.encode(sentences)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):

        sim = util.cos_sim(embeddings[i], embeddings[i-1]).item()

        if sim > similarity_threshold and len(" ".join(current_chunk)) < max_chunk_size:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks