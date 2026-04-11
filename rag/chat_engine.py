from langchain_ollama import OllamaLLM
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
import re



# -----------------------
# Load LLM
# -----------------------

llm = OllamaLLM(
    model="phi3",
    temperature=0,
    num_predict=120
)

# -----------------------
# Main Chat Function
# -----------------------

def ask_ai(question: str) -> None:

    # -----------------------
    # Step 1: Query Rewrite (only if needed)
    # -----------------------

    rewritten_query = question

    if len(question.split()) < 6:
        rewritten_query = rewrite_query(question)

    # -----------------------
    # Step 2: Multi Query Generation
    # -----------------------

    queries = generate_queries(rewritten_query)

    # -----------------------
    # Step 3: Query Decomposition
    # -----------------------

    sub_queries = decompose_query(rewritten_query)

    # Limit total queries
    all_queries = list(set((queries + sub_queries)[:4]))

    # -----------------------
    # Step 4: Parallel Retrieval
    # -----------------------

    docs = []

    def retrieve(q):
        try:
            return compression_retriever.invoke(q)
        except:
            return []

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(retrieve, all_queries)

    for r in results:
        docs.extend(r)

    if len(docs) == 0:
        print("No documents retrieved.")
        return

    # -----------------------
    # Step 5: Remove Duplicate Docs
    # -----------------------

    unique_docs = {}

    for doc in docs:
        if hasattr(doc, "page_content"):
            unique_docs[doc.page_content] = doc
        else:
            unique_docs[str(doc)] = doc

    docs = list(unique_docs.values())

    # -----------------------
    # Step 6: Rerank
    # -----------------------

    scored_docs = rerank(
    rewritten_query + " highest package placement salary offer",
    docs,
    top_k=5,
    return_scores=True
)

    if not scored_docs:
        print("No documents retrieved after reranking.")
        return

    # Calculate average confidence
    avg_score = sum(score for _, score in scored_docs) / len(scored_docs)

    # Adaptive document selection
    if avg_score > 0.75:
        docs = [doc for doc, score in scored_docs[:3]]

    elif avg_score > 0.55:
        docs = [doc for doc, score in scored_docs[:6]]

    else:
        docs = [doc for doc, score in scored_docs[:10]]

    if not scored_docs:
        print("No documents retrieved after reranking.")
        return


    # -----------------------
    # Step 7: Build Context
    # -----------------------

    context = "\n\n".join([doc.page_content for doc in docs])

    MAX_CONTEXT = 1500
    context = context[:MAX_CONTEXT]

    if context.strip() == "":
        print("No relevant context retrieved.")
        return

    # -----------------------
    # Step 8: Extract Sources
    # -----------------------

    sources = set()

    for doc in docs:
        if hasattr(doc, "metadata"):
            sources.add(doc.metadata.get("source", "Unknown"))

    # -----------------------
    # Step 9: Conversation Memory
    # -----------------------

    history = get_recent_history()

    # -----------------------
    # Step 10: Prompt
    # -----------------------

    prompt = f"""
You are a professional AI assistant for Chandigarh University.

RULES:

1. Answer ONLY using the context.
2. If a number (like package/salary) is found → include it in a complete sentence.
3. Keep answer SHORT (1 sentence only).
4. Do NOT add extra explanation.
5. If not found → say exactly:
"I could not find that information in the university knowledge base."

Context:
{context}

Question:
{question}

Answer:
"""

# -----------------------
# Step 11: Stream Response
# -----------------------

    answer = ""

    try:
        print("\nAnswer:\n")

        for chunk in llm.stream(prompt):
            print(chunk, end="", flush=True)
            answer += chunk

    except Exception as e:
        print("LLM error:", e)
        return

    print("\n")

    # -----------------------
    # Step 12: Verify Answer
    # -----------------------

    try:
        is_valid = verify_answer(question, context, answer)

        if not is_valid:
            answer = "I could not find that information in the university knowledge base."
            print("\nAnswer:\n")
            print(answer)
            return

        # -----------------------
        # Step 13: Grounding Check
        # -----------------------

        grounded = is_answer_grounded(answer, context)

        if not grounded:
            answer = "I could not find that information in the university knowledge base."
            print("\nAnswer:\n")
            print(answer)
            return

        # -----------------------
        # Step 14: Numeric Fact Guard
        # -----------------------

        if not numbers_exist_in_context(answer, context):
            answer = "I could not find that information in the university knowledge base."
            print("\nAnswer:\n")
            print(answer)
            return

    except Exception as e:
        print("Verification error:", e)
        answer = "I could not find that information in the university knowledge base."
        print("\nAnswer:\n")
        print(answer)
        return

    # -----------------------
    # Step 15: Print Answer
    # -----------------------

    print("\n" + "="*50)
    print("🎓 Chandigarh University AI Assistant")
    print("="*50)

    print("\nAnswer:\n")
    print(answer.strip())

    print("\n" + "="*50)

    # -----------------------
    # Step 16: Save Memory
    # -----------------------

    
    if "package" in question.lower():

        matches = re.findall(r'(\d+\.?\d*)\s*(crore|cr|lakh|lpa)', context.lower())

        max_value = 0
        best_match = None

        for num, unit in matches:
            num = float(num)

        # Normalize everything to LPA for comparison
            if unit in ["crore", "cr"]:
                value = num * 100   # 1 crore = 100 LPA
            elif unit == "lakh":
                value = num / 100   # lakh → LPA
            else:
                value = num         # already LPA

            if value > max_value:
                max_value = value
                best_match = f"{num} {unit}".upper()

        if best_match:
            answer = f"The highest package in Chandigarh University is {best_match}."

    add_message(question, answer)
    return answer, sources

