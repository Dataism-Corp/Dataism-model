from typing import List, Dict
from urllib.parse import urlparse

def _domain(u: str) -> str:
    try:
        return urlparse(u).netloc or ""
    except Exception:
        return ""

def rrf_blend(lists: List[List[Dict]], k: int = 60, C: int = 60) -> List[Dict]:
    scores = {}
    meta = {}
    for lst in lists:
        for rank, item in enumerate(lst[:k], start=1):
            key = item.get("url") or f"{item.get('title','')}-{rank}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (C + rank)
            if key not in meta:
                meta[key] = item
    fused = [{"score": v, **meta[k]} for k, v in scores.items()]
    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused

def normalize_kb_hits(hits: List[Dict]) -> List[Dict]:
    out = []
    for h in hits:
        m = h.get("metadata", {})
        title = m.get("title","Unknown")
        page = m.get("page",-1)
        sect = m.get("section","")
        cite = f"p.{page}" if page != -1 else sect
        out.append({
            "source_type": "kb",
            "provider": "local_rag",
            "title": title,
            "url": "",
            "text": h.get("document",""),
            "published_at": None,
            "score": float(h.get("rerank_score", 0.0)),
            "kb_citation": {"title": title, "page": page, "section": sect, "cite": cite},
            "raw": h,
        })
    return out

def cap_per_domain(snippets: List[Dict], max_per_domain: int = 2) -> List[Dict]:
    counts = {}
    kept = []
    for s in snippets:
        d = _domain(s.get("url","")) if s.get("url") else "local"
        if counts.get(d, 0) >= max_per_domain:
            continue
        kept.append(s)
        counts[d] = counts.get(d, 0) + 1
    return kept

def enforce_min_chars(snippets: List[Dict], min_chars: int = 200) -> List[Dict]:
    return [s for s in snippets if len((s.get("text") or "").strip()) >= min_chars]
