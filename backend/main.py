from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
from backend.rag.chatbot import chatbot_instance
from backend.ingest import ingest_data

app = FastAPI(title="Enterprise Knowledge Assistant API")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    role: str = "General"
    history: List[Message] = []

class ChatResponse(BaseModel):
    response: str
    suggestions: List[str]

@app.get("/")
def read_root():
    return {"message": "Enterprise Knowledge Assistant API is running."}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        # Convert history from Pydantic models to dicts for the chatbot
        history_dicts = [{"role": msg.role, "content": msg.content} for msg in request.history]
        
        response_text = chatbot_instance.query(request.query, request.role, history_dicts)
        suggestions = chatbot_instance.suggest_questions(request.query)
        
        return ChatResponse(response=response_text, suggestions=suggestions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
def trigger_ingestion(background_tasks: BackgroundTasks):
    """Triggers the document ingestion process in the background."""
    try:
        background_tasks.add_task(ingest_data, "data", "faiss_index")
        return {"message": "Ingestion started in the background."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
