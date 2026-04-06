import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.chat_engine import ask_ai


def admission_agent(question):

    prompt = f"""
You are the University Admission Assistant.

Help students with:

- Admission process
- Eligibility
- Application steps
- Documents required
- Deadlines

Answer clearly for students.

Question: {question}
"""

    return ask_ai(prompt)