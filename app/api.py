# app/api.py
from __future__ import annotations

import os
import glob
import threading
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# ML
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# -------- env / config --------
load_dotenv()  # read .env in the repo root

QWEN_BASE_DIR = os.getenv("QWEN_BASE_DIR")  # e.g. /home/dateria/models/Qwen2.5-14B-Instruct
QWEN_CODE_DIR = os.getenv("QWEN_CODE_DIR")  # e.g. /home/dateria/models/Qwen2.5-Coder-14B
MODEL_DEFAULT = os.getenv("MODEL_DEFAULT", "base")

# Map friendly model names -> local directories
MODEL_PATHS: Dict[str, Optional[str]] = {
    "Qwen2.5-14B-Instruct": QWEN_BASE_DIR,
    "Qwen2.5-Coder-14B": QWEN_CODE_DIR,
}

app = FastAPI(title="Dataism Model API", version="0.1")

# -------- model cache / helpers --------
_models_lock = threading.Lock()
_models: Dict[str, Any] = {}  # name -> HF pipeline

def _validate_model_dir(path: str) -> None:
    if not path or not os.path.isdir(path):
        raise FileNotFoundError(f"Model path not found: {path!r}")
    # quick sanity: need a config and some weights
    has_cfg = os.path.exists(os.path.join(path, "config.json"))
    has_safetensors = bool(glob.glob(os.path.join(path, "model-*.safetensors")))
    has_index = os.path.exists(os.path.join(path, "pytorch_model.bin.index.json"))
    if not has_cfg or not (has_safetensors or has_index):
        raise FileNotFoundError(
            f"Model files missing in {path!r} (need config.json and weights)."
        )

def _load_pipeline(name: str, path: str):
    """Lazy-load and cache a text-generation pipeline for `name`."""
    with _models_lock:
        if name in _models:
            return _models[name]

        _validate_model_dir(path)

        tok = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True, local_files_only=True
        )
        mdl = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype="auto",
            device_map="auto",     # uses GPU if available
            trust_remote_code=True,
            local_files_only=True,
        )
        gen = pipeline(
            task="text-generation",
            model=mdl,
            tokenizer=tok,
            device=0 if torch.cuda.is_available() else -1,
        )
        _models[name] = gen
        return gen

def _available_models() -> List[str]:
    out = []
    for name, path in MODEL_PATHS.items():
        if path and os.path.isdir(path):
            out.append(name)
    return out

# -------- request / response models --------
class GenerateRequest(BaseModel):
    prompt: str
    model: Optional[str] = None  # name in MODEL_PATHS
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9

class GenerateResponse(BaseModel):
    model: str
    output: str

# -------- routes --------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "env_seen": {
            "QWEN_BASE_DIR": QWEN_BASE_DIR,
            "QWEN_CODE_DIR": QWEN_CODE_DIR,
            "MODEL_DEFAULT": MODEL_DEFAULT,
        },
        "available_models": _available_models(),
    }

@app.get("/models")
def models():
    """List model names the API can serve (based on existing folders)."""
    return {"models": _available_models()}

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    # pick model
    name = req.model or (
        "Qwen2.5-14B-Instruct" if "Qwen2.5-14B-Instruct" in _available_models()
        else (_available_models()[0] if _available_models() else None)
    )
    if not name:
        raise HTTPException(status_code=404, detail="No local models are available.")

    path = MODEL_PATHS.get(name)
    if not path:
        raise HTTPException(status_code=404, detail=f"Unknown model name: {name}")

    try:
        gen = _load_pipeline(name, path)
        outputs = gen(
            req.prompt,
            max_new_tokens=req.max_new_tokens,
            do_sample=True,
            temperature=req.temperature,
            top_p=req.top_p,
            eos_token_id=getattr(gen.tokenizer, "eos_token_id", None),
        )
        # HF pipeline returns list[dict]
        text = outputs[0]["generated_text"]
        return {"model": name, "output": text}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")

@app.post("/codegen", response_model=GenerateResponse)
def codegen(req: GenerateRequest):
    """Same as /generate but defaults to the Coder model."""
    if not req.model:
        req.model = "Qwen2.5-Coder-14B"
    return generate(req)
