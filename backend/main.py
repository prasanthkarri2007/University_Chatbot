from fastapi import FastAPI
from pydantic import BaseModel
from rag.chat_engine import ask_ai

app = FastAPI()

class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "University AI Assistant Running"}


@app.post("/chat")
def chat(question: Question):
    response = ask_ai(question.question)
    return {"answer": response["result"]}