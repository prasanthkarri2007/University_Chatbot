def generate_queries(question: str):

    q = question.lower().strip()

    queries = []

    # Base queries
    queries.append(q)
    queries.append(f"chandigarh university {q}")
    queries.append(f"cu mohali {q}")

    # Intent-based expansion (smart 🔥)

    if "package" in q or "placement" in q or "salary" in q:
        queries.append("highest package chandigarh university placement")
        queries.append("placement statistics chandigarh university")

    elif "fees" in q:
        queries.append("chandigarh university fees structure")
        queries.append("course fees chandigarh university")

    elif "hostel" in q:
        queries.append("chandigarh university hostel fees facilities")
        queries.append("cu hostel details")

    elif "admission" in q:
        queries.append("chandigarh university admission process")
        queries.append("eligibility chandigarh university")

    else:
        queries.append(f"{q} details chandigarh university")

    # Remove duplicates + limit
    queries = list(set(queries))

    return queries[:3]   # 🔥 keep it small for speed