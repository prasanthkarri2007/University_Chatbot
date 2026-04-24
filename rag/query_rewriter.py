def rewrite_query(question: str) -> str:

    q = question.lower().strip()

    # Remove unnecessary filler words
    stop_words = [
        "what", "is", "the", "of", "in", "for",
        "tell", "me", "about", "please", "give",
        "details", "can", "you"
    ]

    words = [w for w in q.split() if w not in stop_words]

    # Join back
    rewritten = " ".join(words)

    # Add CU context if missing
    if "chandigarh university" not in rewritten and "cu" not in rewritten:
        rewritten = "chandigarh university " + rewritten

    # Limit to max 8 words
    rewritten = " ".join(rewritten.split()[:8])

    return rewritten