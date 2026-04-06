import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.chat_engine import ask_ai


def notice_agent(question):

    prompt = f"""
You are the University Notice and Announcement Assistant.

Help students with:

- Latest university announcements
- Academic notices
- Event updates
- Important deadlines
- Circulars
- University events

Answer clearly for a student.

Question: {question}
"""

    return ask_ai(prompt)