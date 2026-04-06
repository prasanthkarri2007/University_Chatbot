import json
import os

MEMORY_FILE = "chat_memory.json"


def load_memory():

    if not os.path.exists(MEMORY_FILE):
        return []

    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_memory(memory):

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)


def add_message(question, answer):

    memory = load_memory()

    memory.append({
        "question": question,
        "answer": answer
    })

    # keep only last 20 messages
    memory = memory[-20:]

    save_memory(memory)


def get_recent_history():

    memory = load_memory()

    history_text = ""

    for msg in memory[-5:]:   # last 5 messages only
        history_text += f"User: {msg['question']}\nAssistant: {msg['answer']}\n"

    return history_text