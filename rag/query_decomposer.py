from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="phi3")


def decompose_query(question):

    prompt = f"""
Break the following question into smaller search queries
that help retrieve university information.

Question:
{question}

Return each query on a new line.
Do not number them.
"""

    response = llm.invoke(prompt)

    queries = [q.strip() for q in response.split("\n") if q.strip()]

    return queries