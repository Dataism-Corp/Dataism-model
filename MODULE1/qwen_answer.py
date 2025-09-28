#!/usr/bin/env python3
"""
Use RAG → ask local Qwen → print grounded answer + sources.
Requires: retriever.py, assemble_prompt.py, your local Qwen model dir.
"""

import os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GenerationConfig
from retriever import retrieve
from assemble_prompt import build_prompt

# Point to your local model (override with env QWEN_DIR if needed)
MODEL_DIR = os.environ.get("QWEN_DIR", os.path.expanduser("~/models/Qwen2.5-14B-Instruct"))

assert torch.cuda.is_available(), "CUDA not available"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

_tok = None
_model = None

def get_qwen():
    global _tok, _model
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True, local_files_only=True)
        if _tok.pad_token is None:
            _tok.pad_token = _tok.eos_token
        _tok.padding_side = "left"
    if _model is None:
        cfg = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True)
        attn_impl = getattr(cfg, "_attn_implementation", "eager")
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            device_map={"": 0},
            torch_dtype=DTYPE,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            local_files_only=True,
        )
        _model.generation_config = GenerationConfig.from_model_config(_model.config)
    return _tok, _model

def generate_answer(prompt_text: str, max_new=512, sample=True):
    tok, model = get_qwen()

    # We already built a self-contained prompt; treat it as a single user turn.
    messages = [{"role": "user", "content": prompt_text}]

    if getattr(tok, "chat_template", None):
        input_ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        attn = torch.ones_like(input_ids)
    else:
        enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
        input_ids = enc["input_ids"]
        attn = enc.get("attention_mask", torch.ones_like(input_ids))

    gen = dict(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new,
        use_cache=True,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        do_sample=bool(sample),
    )
    if sample:
        gen.update(dict(temperature=0.8, top_p=0.95, repetition_penalty=1.1, no_repeat_ngram_size=3))

    with torch.inference_mode():
        out = model.generate(**gen)
    new_ids = out[0, input_ids.shape[-1]:]
    return tok.decode(new_ids, skip_special_tokens=True).strip()

def main():
    if len(sys.argv) < 2:
        print("Usage: python qwen_answer.py \"your question here\"")
        sys.exit(1)
    question = sys.argv[1]

    # 1) retrieve KB context
    hits = retrieve(question)
    if not hits:
        print("No relevant context found in the KB.")
        return

    # 2) assemble grounded prompt
    prompt = build_prompt(question, hits)

    # 3) show sources up-front
    print("\n=== CONTEXT SOURCES ===")
    for h in hits:
        m = h["metadata"]
        title = m.get("title", "Unknown")
        page = m.get("page", -1)
        sect = m.get("section", "")
        print(f"- {title} | {'p.'+str(page) if page!=-1 else sect} | score={h.get('rerank_score',0):.3f}")

    # 4) get answer from local Qwen
    print("\n=== ANSWER ===\n")
    ans = generate_answer(prompt)
    print(ans)

if __name__ == "__main__":
    main()
