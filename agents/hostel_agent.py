import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.chat_engine import ask_ai


def hostel_agent(question):

    prompt = f"""
You are the University Hostel Assistant.

Help students with:

- Hostel rules
- Hostel fees
- Mess facilities
- Room types
- Hostel timings
- Hostel application process

Answer clearly for a student.

Question: {question}
"""

    return ask_ai(prompt)