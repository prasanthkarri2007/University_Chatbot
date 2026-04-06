import re


def extract_numbers(text: str):
    """
    Extract numeric values like:
    54 lakh
    1.7 crore
    94%
    500 students
    """

    pattern = r"\d+(\.\d+)?\s?(crore|lakh|%|percent|students|companies)?"

    matches = re.findall(pattern, text.lower())

    numbers = []
    for match in matches:
        num = match[0]
        numbers.append(num)

    return numbers


def numbers_exist_in_context(answer: str, context: str) -> bool:
    """
    Ensures every number in the answer exists in the context.
    """

    answer_numbers = extract_numbers(answer)
    context_numbers = extract_numbers(context)

    for num in answer_numbers:
        if num not in context_numbers:
            return False

    return True