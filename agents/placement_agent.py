import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.chat_engine import ask_ai


def placement_agent(question):

    prompt = f"""
You are the University Placement Assistant.

Help students with:

- Placement statistics
- Companies visiting campus
- Highest package
- Average package
- Internship opportunities
- Placement eligibility rules

Explain clearly for a student.

Question: {question}
"""

    return ask_ai(prompt)