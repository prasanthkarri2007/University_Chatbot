import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.chat_engine import ask_ai


def department_agent(question):

    prompt = f"""
You are the University Department Assistant.

Help students with:

- Department information
- Faculty information
- Programs offered by departments
- Courses in departments
- Department facilities
- Research labs
- Academic programs (B.Tech, M.Tech, etc.)

Answer clearly for a student.

Question: {question}
"""

    return ask_ai(prompt)