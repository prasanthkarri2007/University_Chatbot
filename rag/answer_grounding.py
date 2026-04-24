from sentence_transformers import SentenceTransformer, util
import nltk
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
# -----------------------
# Setup
# -----------------------

try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt")

from nltk.tokenize import sent_tokenize

# Load model once (fast)
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")


# -----------------------
# Grounding Function
# -----------------------

def is_answer_grounded(answer: str, context: str, threshold: float = 0.5) -> bool:
    """
    Checks if answer is grounded in context using sentence similarity.
    Optimized for speed + accuracy.
    """

    if not answer.strip() or not context.strip():
        return False

    answer_sentences = sent_tokenize(answer)
    context_sentences = sent_tokenize(context)

    if not answer_sentences or not context_sentences:
        return False

    # Encode all sentences at once (⚡ FAST)
    answer_embeddings = model.encode(answer_sentences, convert_to_tensor=True)
    context_embeddings = model.encode(context_sentences, convert_to_tensor=True)

    grounded_count = 0

    # Compare each answer sentence
    for i, a_emb in enumerate(answer_embeddings):

        scores = util.cos_sim(a_emb, context_embeddings)[0]
        best_score = float(scores.max())

        # print(f"[DEBUG] Sentence {i} score:", best_score)

        if best_score >= threshold:
            grounded_count += 1

    grounding_ratio = grounded_count / len(answer_sentences)

    return grounding_ratio >= 0.6