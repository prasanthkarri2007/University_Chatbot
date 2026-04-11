def generate_queries(question):

    q = question.lower()

    queries = [
        q,
        f"chandigarh university {q}",
        f"cu mohali {q}",
        f"placement statistics chandigarh university"
    ]

    return list(set(queries))