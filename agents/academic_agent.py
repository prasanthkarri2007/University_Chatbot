import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.chat_engine import ask_ai


def academic_agent(question):

    prompt = f"""
You are an Academic Assistant for the university.

You help students with:
- Course information
- Syllabus
- Attendance rules
- Exam rules
- Academic regulations

Question: {question}
"""

    return ask_ai(prompt)