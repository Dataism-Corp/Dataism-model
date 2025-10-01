from fastapi import FastAPI
from app import api

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/models")
async def list_models():
    # Later we’ll make this dynamic, for now it’s hardcoded
    return {"models": ["Qwen2.5-14B-Instruct", "Qwen2.5-Coder-14B"]}

@app.post("/generate")
async def generate(input_text: str):
    # This will call the api.py logic
    return api.run_model(input_text)
