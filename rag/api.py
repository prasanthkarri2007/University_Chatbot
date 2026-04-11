from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from rag.chat_engine import ask_ai


# -----------------------
# Initialize App
# -----------------------
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(
    title="University AI Assistant API",
    description="API for Chandigarh University AI Assistant",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------
# Enable CORS (for frontend)
# -----------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change later for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------
# Request Schema
# -----------------------

class Query(BaseModel):
    question: str


# -----------------------
# Response Schema
# -----------------------

class ResponseModel(BaseModel):
    answer: str
    sources: list


# -----------------------
# Health Check Route
# -----------------------

@app.get("/")
def home():
    return {"message": "University AI Assistant API is running 🚀"}


# -----------------------
# Main AI Route
# -----------------------

@app.post("/ask", response_model=ResponseModel)
def ask(query: Query):

    try:
        answer, sources = ask_ai(query.question)

        return {
            "answer": answer,
            "sources": list(sources)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )
    
@app.get("/")
def home():
    return {"message": "Chandigarh University AI Assistant API is running"}