import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.chat_engine import ask_ai


def fee_agent(question):

    prompt = f"""
You are the University Fee and Finance Assistant.

Help students with:

- Tuition fee
- Semester fee
- Fee payment process
- Scholarships
- Fee deadlines
- Refund policy

Answer clearly for a student.

Question: {question}
"""

    return ask_ai(prompt)