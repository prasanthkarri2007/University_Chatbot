from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="phi3")

def generate_queries(question):
    prompt = f"""
Generate 4 search queries specifically about Chandigarh University.

The queries must stay focused on Chandigarh University information.

User Question:
{question}

Return one query per line.
Do not number them.
"""
    response = llm.invoke(prompt)

    queries = [q.strip() for q in response.split("\n") if q.strip()]

    return queries