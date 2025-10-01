import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Load model paths from environment
QWEN_BASE_DIR = os.getenv("QWEN_BASE_DIR")
QWEN_CODE_DIR = os.getenv("QWEN_CODE_DIR")

# Register available models
MODELS = {
    "Qwen2.5-14B-Instruct": QWEN_BASE_DIR,
    "Qwen2.5-Coder-14B": QWEN_CODE_DIR
}

# Request schema
class GenerateRequest(BaseModel):
    prompt: str
    model: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/models")
def list_models():
    return {"models": list(MODELS.keys())}

@app.post("/generate")
def generate(req: GenerateRequest):
    if req.model not in MODELS:
        raise HTTPException(status_code=404, detail="Model Not Found")
    # placeholder response (later we’ll load the real model here)
    return {
        "model": req.model,
        "output": f"[fake output from {req.model}] Prompt was: {req.prompt}"
    }
