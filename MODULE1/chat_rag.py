#!/usr/bin/env python3
# chat_rag.py — interactive chat with smart RAG and faster defaults

import os, torch, time, re
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GenerationConfig
from retriever import retrieve, get_reranker  # warmup
from assemble_prompt import build_prompt

MODEL_DIR = os.path.expanduser("~/models/Qwen2.5-14B-Instruct")
assert torch.cuda.is_available(), "CUDA not available"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
USE_RAG = True

def t(): return time.perf_counter()
def log(x): print(x, flush=True)

# ---- heuristics ----
_GREET_PAT = re.compile(r"^(hi|hello|hey|yo|sup|salam|salaam|assalamualaikum|thanks|thank you|ok|okay)\b[.!?]*$", re.I)
_QWORDS = {"what","how","why","when","where","which","who","whom","does","do","is","are","can","should","according"}

def should_use_rag(user: str) -> bool:
    u = user.strip()
    if not u: return False
    if _GREET_PAT.match(u): return False
    if len(u.split()) < 3 and "?" not in u: return False
    if "?" in u: return True
    if any(w in u.lower() for w in _QWORDS): return True
    return False

def dynamic_max_new(user: str, used_rag: bool) -> int:
    # keep it snappy on short prompts
    if not used_rag and (len(user) < 20 or _GREET_PAT.match(user)): return 96
    if len(user) < 80: return 256
    return 512

def pick_attn_impl(cfg):
    # try flash-attn2 if available; else eager
    try:
        from transformers.utils import is_flash_attn_2_available
        if is_flash_attn_2_available():
            return "flash_attention_2"
    except Exception:
        pass
    return "eager"

def has_chat_template(tok):
    return getattr(tok, "chat_template", None) not in (None, "")

def build_inputs(tok, messages, device):
    if has_chat_template(tok):
        input_ids = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        attn = torch.ones_like(input_ids)
        return input_ids, attn, input_ids.shape[-1]

    # Fallback transcript
    sys_txt = ""
    for m in messages:
        if m["role"] == "system":
            sys_txt = m["content"].strip()
    lines = []
    if sys_txt:
        lines.append(f"System: {sys_txt}")
    for m in messages:
        if m["role"] == "user":
            lines.append(f"User: {m['content'].strip()}")
        elif m["role"] == "assistant":
            lines.append(f"Assistant: {m['content'].strip()}")
    lines.append("Assistant:")
    text = "\n".join(lines)

    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
    input_ids = enc["input_ids"]
    attn = enc.get("attention_mask", torch.ones_like(input_ids))
    return input_ids, attn, input_ids.shape[-1]

def generate_reply(model, tok, messages, max_new=512, sample=True):
    device = next(model.parameters()).device
    input_ids, attention_mask, prompt_len = build_inputs(tok, messages, device)

    model.generation_config = GenerationConfig.from_model_config(model.config)
    gen = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new,
        use_cache=True,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        do_sample=bool(sample),
    )
    if sample:
        gen.update(dict(temperature=0.7, top_p=0.9, repetition_penalty=1.05))

    torch.cuda.synchronize(); a = t()
    with torch.inference_mode():
        out = model.generate(**gen)
    torch.cuda.synchronize(); b = t()

    new_ids = out[0, prompt_len:]
    reply = tok.decode(new_ids, skip_special_tokens=True).strip()
    return reply, int(new_ids.shape[-1]), (b - a)

def main():
    # Warm up reranker once so it doesn't download on first user turn
    try:
        get_reranker()
    except Exception:
        pass

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    cfg = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True)
    attn_impl = pick_attn_impl(cfg)

    log(f"GPU: {torch.cuda.get_device_name(0)} | Torch: {torch.__version__} | attn: {attn_impl}")
    a = t()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map={"": 0},
        torch_dtype=DTYPE,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        local_files_only=True,
    )
    model.generation_config = GenerationConfig.from_model_config(model.config)
    torch.cuda.synchronize(); b = t()
    log(f"✓ Model loaded in {b-a:.2f}s")

    history = [{"role": "system", "content": "You are a helpful assistant."}]
    print("\nType 'exit' to quit.\n")

    while True:
        user = input("You: ").strip()
        if not user or user.lower() in ("exit", "quit"): break

        use_rag_now = USE_RAG and should_use_rag(user)
        hits = None
        user_payload = user

        if use_rag_now:
            hits = retrieve(user)
            # If nothing confidently relevant, fall back to plain chat
            if hits:
                user_payload = build_prompt(user, hits)
            else:
                use_rag_now = False

        messages_in = history + [{"role": "user", "content": user_payload}]
        max_new = dynamic_max_new(user, use_rag_now)

        reply, new, dt = generate_reply(model, tok, messages_in, max_new=max_new, sample=True)
        print(f"Assistant: {reply}")
        print(f"[{new} toks in {dt:.2f}s  ~{new/max(dt,1e-6):.1f} tok/s]")

        if use_rag_now and hits:
            print("Sources:")
            for h in hits:
                m = h["metadata"]
                title = m.get("title","Unknown")
                page = m.get("page",-1)
                sect = m.get("section","")
                print(f" • {title} | {'p.'+str(page) if page!=-1 else sect} | score={h.get('rerank_score',0):.3f}")
        print()

        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    main()
