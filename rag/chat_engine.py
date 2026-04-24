from rag.memory import add_message, get_recent_history
from rag.reranker import rerank
from rag.query_rewriter import rewrite_query
from rag.context_compressor import compression_retriever
from rag.multi_query import generate_queries
from rag.query_decomposer import decompose_query
from rag.context_filter import filter_sentences
from rag.answer_verifier import verify_answer
from rag.answer_grounding import is_answer_grounded
from rag.number_guard import numbers_exist_in_context

from concurrent.futures import ThreadPoolExecutor
from langchain_groq import ChatGroq
import os
import re


# -----------------------
# Load LLM (Groq)
# -----------------------

llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


# -----------------------
# Main Chat Function
# -----------------------

def ask_ai(question: str):

    # -----------------------
    # Step 1: Query Rewrite
    # -----------------------

    rewritten_query = question
    if len(question.split()) < 6:
        rewritten_query = rewrite_query(question)

    # -----------------------
    # Step 2: Multi Query
    # -----------------------

    queries = generate_queries(rewritten_query)

    # -----------------------
    # Step 3: Decompose Query
    # -----------------------

    sub_queries = decompose_query(rewritten_query)

    all_queries = list(set((queries + sub_queries)[:3]))  # reduced for speed

    # -----------------------
    # Step 4: Parallel Retrieval
    # -----------------------

    docs = []

    def retrieve(q):
        try:
            return compression_retriever.invoke(q)
        except:
            return []

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = executor.map(retrieve, all_queries)

    for r in results:
        docs.extend(r)

    if not docs:
        return "I could not find that information in the university knowledge base.", []

    # -----------------------
    # Step 5: Remove Duplicates
    # -----------------------

    unique_docs = {}

    for doc in docs:
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        unique_docs[content] = doc

    docs = list(unique_docs.values())

    # -----------------------
    # Step 6: Rerank
    # -----------------------

    scored_docs = rerank(
        rewritten_query,
        docs,
        top_k=5,
        return_scores=True
    )

    if not scored_docs:
        return "I could not find that information in the university knowledge base.", []

    avg_score = sum(score for _, score in scored_docs) / len(scored_docs)

    if avg_score > 0.75:
        docs = [doc for doc, _ in scored_docs[:3]]
    elif avg_score > 0.55:
        docs = [doc for doc, _ in scored_docs[:5]]
    else:
        docs = [doc for doc, _ in scored_docs[:7]]

    # -----------------------
    # Step 7: Build Context
    # -----------------------

    context = filter_sentences(rewritten_query, docs)

    context = context[:1500]

    if not context.strip():
        return "I could not find that information in the university knowledge base.", []

    # -----------------------
    # Step 8: Extract Sources
    # -----------------------

    sources = set()
    for doc in docs:
        if hasattr(doc, "metadata"):
            sources.add(doc.metadata.get("source", "Unknown"))

    # -----------------------
    # Step 9: Prompt
    # -----------------------

    prompt = f"""
You are a smart and helpful AI assistant for Chandigarh University.

Answer the question using ONLY the provided context.

Rules:
- Give a clear and complete sentence
- Be natural and human-like
- Do NOT just return numbers
- If answer is not found, say:
  "I don't have that information based on the available data."

Context:
{context}

Question:
{question}

Answer:
"""

    # -----------------------
    # Step 10: LLM Call (Groq)
    # -----------------------

    try:
        answer = llm.invoke(prompt).content.strip()
    except Exception as e:
        print("LLM error:", e)
        return "Error generating answer.", []

    # -----------------------
    # Step 11: Verification
    # -----------------------

    try:
        if not verify_answer(question, context, answer):
            return "I could not find that information in the university knowledge base.", []

        if not is_answer_grounded(answer, context):
            return "I could not find that information in the university knowledge base.", []

        if not numbers_exist_in_context(answer, context):
            return "I could not find that information in the university knowledge base.", []

    except Exception as e:
        print("Verification error:", e)
        return "I could not find that information in the university knowledge base.", []

    # -----------------------
    # Step 12: Smart Package Fix 🔥
    # -----------------------

    if "package" in question.lower():

        matches = re.findall(r'(\d+\.?\d*)\s*(crore|cr|lakh|lpa)', context.lower())

        max_value = 0
        best_match = None

        for num, unit in matches:
            num = float(num)

            if unit in ["crore", "cr"]:
                value = num * 100
            elif unit == "lakh":
                value = num / 100
            else:
                value = num

            if value > max_value:
                max_value = value
                best_match = f"{num} {unit}".upper()

        if best_match:
            answer = f"The highest package in Chandigarh University is {best_match}."

    # -----------------------
    # Step 13: Save Memory
    # -----------------------

    add_message(question, answer)

    # -----------------------
    # Step 14: Return
    # -----------------------

    return answer, list(sources)