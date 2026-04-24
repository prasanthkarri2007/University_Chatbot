def decompose_query(question: str):

    q = question.lower().strip()

    queries = []

    # Split based on common connectors
    if " and " in q:
        parts = q.split(" and ")
    elif " or " in q:
        parts = q.split(" or ")
    elif "," in q:
        parts = q.split(",")
    else:
        parts = [q]

    # Clean parts
    for part in parts:
        part = part.strip()
        if part:
            queries.append(part)

    # Add CU context if missing
    final_queries = []

    for q in queries:
        if "chandigarh university" not in q and "cu" not in q:
            final_queries.append(f"chandigarh university {q}")
        else:
            final_queries.append(q)

    return list(set(final_queries))[:3]