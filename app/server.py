from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# Example input model
class Query(BaseModel):
    text: str

# Example model run endpoint
@app.post("/run")
def run_model(query: Query):
    # For now just echo the input text
    # Later this will call your real model logic
    return {"response": f"You sent: {query.text}"}
