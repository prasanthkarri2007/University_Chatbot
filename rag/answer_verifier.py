from langchain_ollama import OllamaLLM

verifier_llm = OllamaLLM(model="phi3")


def verify_answer(question, context, answer):

    prompt = f"""
You are a strict fact-checking AI.

Check whether the answer is fully supported by the provided context.

Question:
{question}

Context:
{context}

Answer:
{answer}

Rules:
- If the answer is supported by the context, respond ONLY with: VALID
- If the answer contains information not present in the context, respond ONLY with: INVALID
"""

    result = verifier_llm.invoke(prompt)

    return "VALID" in result.upper()