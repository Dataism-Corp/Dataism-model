from typing import List, Dict
import textwrap

SYSTEM_RULES = """You are a grounded assistant. Use only the provided context to answer. Cite sources at the end like [Title p.3] or [Title §chunk_2]. If information is missing, say so."""

def build_prompt(query: str, contexts: List[Dict], max_context_chars: int = 14000) -> str:
    header_blocks = []
    body_blocks = []
    total = 0

    for c in contexts:
        meta = c["metadata"]
        title = meta.get("title","Unknown")
        page = meta.get("page",-1)
        sect = meta.get("section","")
        head = f"### {title} — " + (f"page {page}" if page!=-1 else sect)
        text = c["document"].strip()
        block = f"{head}\n{text}\n"
        if total + len(block) > max_context_chars:
            break
        header_blocks.append(head)
        body_blocks.append(block)
        total += len(block)

    citations = []
    for c in contexts:
        m = c["metadata"]
        title = m.get("title","Unknown")
        if m.get("page",-1) != -1:
            citations.append(f"[{title} p.{m['page']}]")
        else:
            citations.append(f"[{title} {m.get('section','')}]")
    citations = list(dict.fromkeys(citations))

    context_str = "\n\n".join(body_blocks)
    cite_str = " ".join(citations) if citations else "(no citations)"
    user_block = textwrap.dedent(f"""
    USER QUESTION:
    {query}

    CONTEXT:
    {context_str}

    Cite sources at the end. If context is insufficient, explicitly state that and do not guess.
    """).strip()

    full = f"<SYSTEM>\n{SYSTEM_RULES}\n</SYSTEM>\n\n{user_block}\n\n[CITATIONS] {cite_str}\n"
    return full
