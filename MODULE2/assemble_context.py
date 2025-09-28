from typing import List, Dict

SYSTEM_RULES = """You are a grounded assistant. Use only the provided context to answer. Cite sources at the end:
- For local docs: [Title p.N] or [Title §chunk_k]
- For web or Wikipedia: [Title (URL, YYYY-MM-DD)]
If information is missing or uncertain, say so and do not guess.""".strip()

def _cite(item: Dict) -> str:
    if item.get("source_type") == "kb":
        kc = item.get("kb_citation") or {}
        title = kc.get("title","Unknown")
        cite = kc.get("cite","")
        return f"[{title} {cite}]"
    else:
        title = item.get("title","Unknown")
        url = item.get("url","")
        date = (item.get("published_at") or "").split("T")[0] if item.get("published_at") else ""
        if date:
            return f"[{title} ({url}, {date})]"
        return f"[{title} ({url})]"

def build_blended_prompt(question: str, snippets: List[Dict], max_context_chars: int = 14000) -> str:
    blocks = []
    total = 0
    for s in snippets:
        header = f"### {s.get('title','Unknown')} — {s.get('source_type')}"
        if s.get("source_type") == "kb":
            kc = s.get("kb_citation") or {}
            cite = kc.get("cite","")
            header += f" ({cite})"
        else:
            if s.get("url"):
                header += f" ({s.get('url')})"
        body = (s.get("text") or "").strip()
        block = f"{header}\n{body}\n"
        if total + len(block) > max_context_chars:
            break
        total += len(block)
        blocks.append(block)

    citations = " ".join([_cite(s) for s in snippets]) if snippets else "(no citations)"

    user_block = f"""    USER QUESTION:
{question}

CONTEXT:
{''.join(blocks)}

Cite sources at the end. If context is insufficient, explicitly state that and do not guess.
""".strip()

    full = f"<SYSTEM>\n{SYSTEM_RULES}\n</SYSTEM>\n\n{user_block}\n\n[CITATIONS] {citations}\n"
    return full
