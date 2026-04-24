from langchain_groq import ChatGroq
import os

# Load verifier LLM
verifier_llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


def verify_answer(question: str, context: str, answer: str) -> bool:

    prompt = f"""
You are a strict fact-checking AI.

Check whether the answer is fully supported by the provided context.

Rules:
- Respond ONLY with VALID or INVALID
- Do NOT explain anything

Question:
{question}

Context:
{context}

Answer:
{answer}
"""

    try:
        result = verifier_llm.invoke(prompt).content.strip().upper()

        return result == "VALID"

    except Exception:
        return False