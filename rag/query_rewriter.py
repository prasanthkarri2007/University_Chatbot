from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="phi3")

def rewrite_query(question):

    prompt = f"""
Convert the following student question into a short search query.

Rules:
- Maximum 8 words
- No explanation
- Only return the search query

Question:
{question}

Search Query:
"""

    try:
        rewritten = llm.invoke(prompt).strip()

        # safety check
        if len(rewritten.split()) > 8:
            return question

        return rewritten

    except:
        return question