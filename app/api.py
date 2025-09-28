
# app/api.py
# Simple REST API for Phase 2 foundation

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

# import core chat + memory helpers
from .core import chat, get_memory  # type: ignore

app = FastAPI(title="Dateria Core API", version="0.1.0")

class ChatIn(BaseModel):
    message: str

class ChatOut(BaseModel):
    reply: str

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}

@app.post("/chat", response_model=ChatOut)
def chat_endpoint(body: ChatIn):
    reply = chat(body.message)
    return {"reply": reply}

@app.get("/memory")
def memory() -> List[Dict[str, Any]]:
    return get_memory()

@app.post("/reset")
def reset_memory():
    """
    Clears conversation memory.
    If reset function exists in core, use it; otherwise fall back to clearing list.
    """
    try:
        # optional: if you add reset_memory() in core.py later
        from .core import reset_memory as do_reset  # type: ignore
        do_reset()
    except Exception:
        # fallback: try to clear the list directly
        from .core import conversation_history  # type: ignore
        conversation_history.clear()
    return {"status": "cleared"}
