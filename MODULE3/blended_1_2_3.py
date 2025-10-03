#!/usr/bin/env python3
# Blended chat: Module 1 (KB/RAG) + Module 2 (Web/Wiki/YouTube) + Module 3 (Memory)
# Web is ONLY used when the user explicitly types "web: ...".
# "docs: ..." forces local KB only. Otherwise: Memory + Local KB (+ chat fallback).
# Memory: only injects relevant facts into the prompt; preferences (e.g., bullets) always influence style.

import os, sys, re, time, yaml, torch
from typing import List, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GenerationConfig

# ---------- Make module paths importable ----------
ROOT = os.path.abspath(os.path.dirname(__file__))
M1 = os.path.join(ROOT, "module1")
M2 = os.path.join(ROOT, "module2")
M3 = os.path.join(ROOT, "module3")
for p in (ROOT, M1, M2, M3):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------- Module 1 (local KB/RAG) ----------
from module1.retriever import retrieve, get_reranker
from module1.assemble_prompt import build_prompt

# ---------- Module 2 (web/Wiki/YouTube) ----------
from module2.orchestrator.blender import (
    rrf_blend, cap_per_domain, enforce_min_chars, normalize_kb_hits
)
from module2.adapters.web.tavily_adapter import search_web_tavily
from module2.adapters.wikipedia.wiki_adapter import wiki_lookup
from module2.adapters.youtube.yt_adapter import yt_transcript
from module2.assemble_context import build_blended_prompt

# ---------- Module 3 (memory) ----------
from module3.core import retriever as m3_retriever, writer as m3_writer, controls as m3_controls

# ---------- Configs ----------
CFG1 = yaml.safe_load(open(os.path.join(M1, "config.yaml"), "r"))
CFG2 = yaml.safe_load(open(os.path.join(M2, "config", "module2.yaml"), "r"))
CFG3 = yaml.safe_load(open(os.path.join(M3, "config", "module3.yaml"), "r"))

# ---------- Model setup ----------
MODEL_DIR = os.path.expanduser("~/models/Qwen2.5-14B-Instruct")
assert torch.cuda.is_available(), "CUDA not available"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def t(): return time.perf_counter()
def log(x): print(x, flush=True)

_GREET_PAT = re.compile(r"^(hi|hello|hey|yo|sup|salam|salaam|assalamualaikum|thanks|thank you|ok|okay)\b[.!?]*$", re.I)
_QWORDS = {"what","how","why","when","where","which","who","whom","does","do","is","are","can","should","according"}

# Small-talk blocker
_SMALLTALK_PAT = re.compile(
    r"""^(
        hi|hello|hey|yo|sup|
        salam|salaam|assalamualaikum|
        (how\s+are\s+you)|howdy|
        what's\s+up|whats\s+up|
        good\s+(morning|evening|afternoon)|
        thanks|thank\s+you|ok|okay
    )\b[.!?]*$""",
    re.IGNORECASE | re.VERBOSE
)

def should_use_rag(user: str) -> bool:
    """Decide if we should hit local KB (not web)."""
    u = (user or "").strip()
    if not u:
        return False
    if _SMALLTALK_PAT.match(u):
        return False
    if len(u.split()) < 3 and "?" not in u:
        return False
    ul = u.lower()
    if "?" in ul:
        return True
    if any(re.search(rf"\b{re.escape(w)}\b", ul) for w in _QWORDS):
        return True
    return False

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
    if sys_txt: lines.append(f"System: {sys_txt}")
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
        input_ids=input_ids, attention_mask=attention_mask,
        max_new_tokens=max_new, use_cache=True,
        pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
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

# ---------- Module 2 helpers ----------
def web_search(query: str) -> List[Dict]:
    w = CFG2.get("web", {})
    return search_web_tavily(
        query=query,
        api_key=os.environ.get(w.get("api_key_env","TAVILY_API_KEY")),
        max_results=int(w.get("max_results", 3)),
        recency_days=w.get("recency_days"),
        timeout_seconds=int(w.get("timeout_seconds", 6)),
        allow_domains=w.get("allow_domains") or None,
        deny_domains=w.get("deny_domains") or None,
    )

def maybe_wiki(question: str) -> List[Dict]:
    if not CFG2.get("wikipedia", {}).get("enabled", False):
        return []
    topic = None
    if question.lower().startswith("wikipedia:"):
        topic = question.split(":",1)[1].strip()
    elif question.istitle() or ("(" in question and ")" in question):
        topic = question.strip()
    if not topic: return []
    obj = wiki_lookup(topic, ttl_hours=CFG2.get("wikipedia",{}).get("cache_ttl_hours",24))
    if not obj: return []
    return [{
        "source_type": "web",
        "provider": "wikipedia",
        "title": obj["title"],
        "url": obj["url"],
        "text": (obj.get("summary") or "")[:2000],
        "published_at": None,
        "score": 0.0,
        "raw": obj,
    }]

def maybe_youtube(question: str) -> List[Dict]:
    y = CFG2.get("youtube", {})
    if not y.get("enabled", False):
        return []
    m = re.search(r"(https?://(?:www\.)?(?:youtube\.com/watch\?v=[\w-]+|youtu\.be/[\w-]+))", question)
    if not m: return []
    url = m.group(1)
    res = yt_transcript(
        url,
        prefer_captions=y.get("prefer_captions", True),
        whisper_model=y.get("whisper_model", "medium"),
        max_duration_minutes=y.get("max_duration_minutes", 120),
        compute_type=y.get("compute_type", "auto"),
        beam_size=y.get("beam_size", 5),
        vad_filter=y.get("vad_filter", True),
        language=y.get("language", "en"),
        tmp_dir=y.get("tmp_dir", "/tmp"),
        audio_format=y.get("audio_format", "m4a"),
    )
    if not res or not res.get("transcript"): return []
    return [{
        "source_type": "web",
        "provider": "youtube",
        "title": f"YouTube transcript {res['video_id']}",
        "url": res["url"],
        "text": res["transcript"][:4000],
        "published_at": None,
        "score": 0.0,
        "raw": res,
    }]

def explicit_overrides(raw: str) -> Tuple[str,str]:
    s = raw.strip()
    if s.lower().startswith("docs:"):
        return "docs", s.split(":",1)[1].strip()
    if s.lower().startswith("web:"):
        return "web", s.split(":",1)[1].strip()
    return "auto", s

def build_sources_line(kb_hits: List[Dict], web_used: List[Dict]) -> str:
    lines = []
    for h in kb_hits or []:
        m = h.get("metadata", {})
        title = m.get("title","Unknown")
        page = m.get("page",-1)
        sect = m.get("section","")
        lines.append(f" • {title} | {'p.'+str(page) if page!=-1 else sect} | score={h.get('rerank_score',0):.3f}")
    for s in web_used or []:
        title = s.get("title","Unknown")
        url = s.get("url","")
        lines.append(f" • {title} — {url}")
    return "\n".join(lines) if lines else "(no sources)"

# ---------- Memory helpers ----------
def render_memory_block(cards: List[Dict]) -> str:
    if not (cards and CFG3.get("enabled", True)):
        return ""
    lines = ["[MEMORY CONTEXT]"]
    for m in cards[: int(CFG3.get("top_k_recall", 8))]:
        kind = (m.get("type") or "fact").capitalize()
        subj = m.get("subject","")
        val  = m.get("value","")
        lines.append(f"• {kind}: {subj} — {val}")
    return "\n".join(lines)

def _relevant_memories(cards: List[Dict], question: str, min_score: float = 0.60, max_keep: int = 8) -> List[Dict]:
    """
    Keep only memories that are likely relevant to the current question.
    - Always allow 'preference' memories (e.g., answer_style).
    - For 'fact'/'note' memories: require either semantic score >= min_score
      OR lexical overlap (shared 3+ letter word).
    """
    if not cards:
        return []

    q_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", (question or "").lower()))

    def is_relevant(c: Dict) -> bool:
        md = c.get("metadata", {})
        ctype = md.get("type", "")
        if ctype == "preference":
            return True
        score = float(c.get("score", 0.0))
        if score >= min_score:
            return True
        text = " ".join([
            c.get("document", ""),
            md.get("type",""), md.get("subject",""), md.get("value","")
        ]).lower()
        return any(w in text for w in q_words)

    filtered = [c for c in cards if is_relevant(c)]

    def sort_key(c):
        md = c.get("metadata", {})
        return (float(c.get("score", 0.0)), float(md.get("_stored_at", 0.0)))

    filtered.sort(key=sort_key, reverse=True)
    return filtered[:max_keep]

# ---------- Main ----------
def main():
    # Warm up reranker
    try:
        get_reranker()
    except Exception:
        pass

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    cfg = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True)
    attn_impl = "eager"
    try:
        from transformers.utils import is_flash_attn_2_available
        if is_flash_attn_2_available():
            attn_impl = "flash_attention_2"
    except Exception:
        pass

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

    print("\nType 'exit' to quit. Use 'web: …' to force web, 'docs: …' to force local KB. Memory commands: /memory list | /memory export\n")

    while True:
        user = input("You: ").strip()
        if not user or user.lower() in ("exit","quit"): break

        # --- Memory controls (Module 3) ---
        if user.strip().lower().startswith("/memory"):
            norm_cmd = re.sub(r"\s+", " ", user).strip()
            print(m3_controls.handle(norm_cmd, CFG3)); print(); continue

        mode, question = explicit_overrides(user)

        # Greeting shortcut (no RAG, no memory prepend)
        if _GREET_PAT.match(question):
            print("Assistant: Hi! Ask a question, or try 'web: latest AI regulation updates this month'.\n")
            continue

        # --- Memory recall (Module 3, pre-context) ---
        mem_cards_raw = m3_retriever.recall(question, CFG3) if CFG3.get("enabled", True) else []

        # Keep only relevant memories (always keep "preference")
        mem_cards = _relevant_memories(
            mem_cards_raw,
            question,
            min_score=float(CFG3.get("min_score", 0.60)),
            max_keep=int(CFG3.get("top_k_recall", 8)),
        )

        # Render memory block (but we’ll only prepend for real queries later)
        mem_block = render_memory_block(mem_cards)

        # Detect bullet preference for formatting later
        has_bullet_pref = any(
            (c.get("type") == "preference" and c.get("subject") == "answer_style" and "bullet" in c.get("value","").lower())
            for c in mem_cards
        )

        # 1) Decide KB retrieval (local only)
        use_rag_now = should_use_rag(question) or (mode in ("docs","web"))

        kb_hits: List[Dict] = []
        if use_rag_now:
            kb_hits = retrieve(question) or []

        # 2) Decide web — ONLY if user typed "web:"
        explicit_web = (mode == "web")
        explicit_docs = (mode == "docs")
        use_web = explicit_web
        if explicit_docs:
            use_web = False

        # 3) Build context (Memory + KB/Web)
        kb_norm = normalize_kb_hits(kb_hits) if kb_hits else []
        web_snips: List[Dict] = []

        if use_web:
            web_snips += web_search(question)
            if question.lower().startswith("wikipedia:"):
                web_snips += maybe_wiki(question)
            if re.search(r"(https?://(?:www\.)?(?:youtube\.com/watch\?v=[\w-]+|youtu\.be/[\w-]+))", question):
                web_snips += maybe_youtube(question)

        if kb_norm or web_snips:
            fused = rrf_blend(
                ([kb_norm] if kb_norm else []) + ([web_snips] if web_snips else []),
                k=60, C=60
            )
            fused = enforce_min_chars(fused, CFG2["orchestration"]["blend"]["min_snippet_chars"])
            fused = cap_per_domain(fused, CFG2["orchestration"]["blend"]["max_per_domain"])
            blended = fused[:CFG2["orchestration"]["blend"]["max_snippets_total"]]
            payload = build_blended_prompt(question, blended)
        elif kb_hits:
            payload = build_prompt(question, kb_hits)
        else:
            payload = question

        # Prepend memory context only for real queries (not small talk)
        if mem_block and ("?" in question or len(question.split()) >= 4):
            user_payload = mem_block + "\n\n" + payload
        else:
            user_payload = payload

        # 4) Generate (short for statements/small talk) + honor bullet preference
        system_txt = "You are a helpful assistant."
        if has_bullet_pref:
            system_txt += " When possible, format answers as concise bullet points."

        messages_in = [
            {"role":"system","content": system_txt},
            {"role":"user","content": user_payload}
        ]
        if not use_rag_now and (len(question) < 20 or _SMALLTALK_PAT.match(question)):
            max_new = 64
        elif "?" not in question:
            max_new = 128
        elif len(question) < 80:
            max_new = 256
        else:
            max_new = 512

        reply, new, dt = generate_reply(model, tok, messages_in, max_new=max_new, sample=True)
        print(f"Assistant: {reply}")
        print(f"[{new} toks in {dt:.2f}s  ~{new/max(dt,1e-6):.1f} tok/s]")

        # 5) Sources (KB/Web)
        if kb_hits or web_snips:
            print("Sources:")
            print(build_sources_line(kb_hits, web_snips))
        print()

        # --- Memory write (Module 3, post-answer) ---
        try:
            m3_writer.write({"user": user, "assistant": reply, "sources": (kb_hits or [])}, CFG3)
        except Exception:
            pass

if __name__ == "__main__":
    main()
