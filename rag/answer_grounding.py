from sentence_transformers import SentenceTransformer, util
import nltk

# Download tokenizer if not already present
try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt")

from nltk.tokenize import sent_tokenize

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def sentence_similarity(a, b):
    emb1 = model.encode(a, convert_to_tensor=True)
    emb2 = model.encode(b, convert_to_tensor=True)

    score = util.cos_sim(emb1, emb2)

    return score.item()


def is_answer_grounded(answer: str, context: str, threshold: float = 0.45) -> bool:
    """
    Checks each sentence in the answer against the context.
    If any sentence is unsupported → answer rejected.
    """

    answer_sentences = sent_tokenize(answer)
    context_sentences = sent_tokenize(context)

    grounded_sentences = 0

    for a_sent in answer_sentences:

        best_score = 0

        for c_sent in context_sentences:
            score = sentence_similarity(a_sent, c_sent)

            if score > best_score:
                best_score = score

        # print(f"Sentence grounding score: {best_score}")

        if best_score >= threshold:
            grounded_sentences += 1

    grounding_ratio = grounded_sentences / len(answer_sentences)

    # print(f"Grounded sentences: {grounded_sentences}/{len(answer_sentences)}")

    return grounding_ratio >= 0.5