import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.chat_engine import ask_ai


def campus_agent(question):

    prompt = f"""
You are the University Campus Facilities Assistant.

Help students with information about:

- Library
- Computer labs
- Research labs
- Sports facilities
- Campus WiFi
- Cafeteria
- Medical facilities
- Transport services
- Campus infrastructure

Answer clearly for a student.

Question: {question}
"""

    return ask_ai(prompt)