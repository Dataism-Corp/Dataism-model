# module3/core/retriever.py
from __future__ import annotations
from typing import Dict, List, Tuple
from .store import search

def _key(md: Dict) -> Tuple[str, str, str]:
    return (
        (md or {}).get("type", ""),
        (md or {}).get("subject", ""),
        (md or {}).get("value", ""),
    )

def _to_card(md: Dict) -> Dict:
    """Convert store metadata back to a 'card'-like dict for prompting."""
    return {
        "type": md.get("type", ""),
        "subject": md.get("subject", ""),
        "value": md.get("value", ""),
        "evidence": md.get("evidence", ""),
        "timestamp": md.get("timestamp", ""),
    }

def recall(query: str, cfg: Dict) -> List[Dict]:
    """
    Recall memory cards using semantic + recency search from the store.
    - Pull a bit more than top_k to allow dedupe.
    - Deduplicate by (type, subject, value) keeping the highest score.
    - Return as normalized 'card' dicts.
    """
    k = int(cfg.get("top_k_recall", 8))
    hits = search(query=query, cfg=cfg, top_k=max(k * 3, 24))  # grab extra for dedupe

    # Dedup by (type, subject, value) keeping best score
    best = {}
    for h in hits or []:
        md = h.get("metadata", {})
        key = _key(md)
        if key not in best or float(h.get("score", 0.0)) > float(best[key].get("score", 0.0)):
            best[key] = h

    # Sort by score desc and take top_k
    keep = sorted(best.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)[:k]

    # Convert back to lightweight 'card' dicts for prompt assembly
    return [_to_card(h.get("metadata", {})) for h in keep]
