# module3/core/store.py
"""
Minimal persistent memory store for Module 3.
- Chroma DB collection: 'memories'
- save_cards(cards, cfg): upsert memory cards
- search(query, cfg, top_k=8): semantic + recency recall (blank query -> recent)
- export_jsonl(out_path, cfg): dump all cards to JSONL
- list_recent(cfg, limit=10): convenience for '/memory list'
"""

from __future__ import annotations
import os, json, time, hashlib
from typing import Dict, List

import chromadb
from chromadb.config import Settings


# ---------- Paths / client ----------
def _index_dir(cfg: Dict) -> str:
    base = cfg.get("paths", {}).get("index_dir") or os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "index"
    )
    os.makedirs(base, exist_ok=True)
    return base

def _client(cfg: Dict) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=_index_dir(cfg), settings=Settings())

def _collection(cfg: Dict):
    return _client(cfg).get_or_create_collection("memories")


# ---------- Helpers ----------
def _now() -> float:
    return time.time()

def _card_id(card: Dict) -> str:
    key = f"{card.get('type','')}-{card.get('subject','')}-{card.get('value','')}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:24]

def _as_text(card: Dict) -> str:
    t = card.get("type", "")
    s = card.get("subject", "")
    v = card.get("value", "")
    return f"{t}: {s} — {v}"


# ---------- Public API ----------
def save_cards(cards: List[Dict], cfg: Dict) -> int:
    if not cards:
        return 0
    col = _collection(cfg)

    ids, docs, metas = [], [], []
    for c in cards:
        cid = _card_id(c)
        ids.append(cid)
        docs.append(_as_text(c))
        metas.append({
            "type": c.get("type", ""),
            "subject": c.get("subject", ""),
            "value": c.get("value", ""),
            "evidence": c.get("evidence", ""),
            "timestamp": c.get("timestamp", ""),
            "_stored_at": _now(),
        })
    col.upsert(ids=ids, documents=docs, metadatas=metas)
    return len(ids)


def search(query: str, cfg: Dict, top_k: int = 8) -> List[Dict]:
    """
    Semantic + recency recall.
    - Blank query: return most recent items.
    - Else: query + light recency boost.
    """
    col = _collection(cfg)

    try:
        if not col.count():
            return []
    except Exception:
        return []

    q = (query or "").strip()
    if not q:
        # NOTE: do NOT include "ids" in the include list (Chroma rejects it there).
        res = col.get(limit=top_k, include=["metadatas", "documents"])
        items = []
        for i in range(len(res["ids"])):
            items.append({
                "id": res["ids"][i],
                "document": res["documents"][i],
                "metadata": res["metadatas"][i],
                "score": 1.0
            })
        items.sort(key=lambda x: float(x["metadata"].get("_stored_at", 0.0)), reverse=True)
        return items[:top_k]

    res = col.query(
        query_texts=[q],
        n_results=max(top_k * 2, 8),
        include=["documents", "metadatas", "distances"],
    )

    items: List[Dict] = []
    for i in range(len(res["ids"][0])):
        md = res["metadatas"][0][i]
        distance = float(res["distances"][0][i])
        base = 1.0 / (1.0 + distance)
        age = max(1.0, _now() - float(md.get("_stored_at", 0.0)))
        rec = min(0.15, 0.15 / (1.0 + age / (60 * 60 * 24)))  # day-scale boost
        items.append({
            "id": res["ids"][0][i],
            "document": res["documents"][0][i],
            "metadata": md,
            "score": base + rec,
        })

    seen = {}
    for it in items:
        key = (it["metadata"].get("type",""), it["metadata"].get("subject",""), it["metadata"].get("value",""))
        if key not in seen or it["score"] > seen[key]["score"]:
            seen[key] = it
    deduped = list(seen.values())
    deduped.sort(key=lambda x: x["score"], reverse=True)
    return deduped[:top_k]


def list_recent(cfg: Dict, limit: int = 10) -> List[Dict]:
    col = _collection(cfg)
    res = col.get(limit=max(limit, 1), include=["metadatas", "documents"])  # no "ids" in include
    items = []
    for i in range(len(res["ids"])):
        items.append({
            "id": res["ids"][i],
            "document": res["documents"][i],
            "metadata": res["metadatas"][i],
            "score": 1.0,
        })
    items.sort(key=lambda x: float(x["metadata"].get("_stored_at", 0.0)), reverse=True)
    return items[:limit]


def export_jsonl(out_path: str, cfg: Dict) -> str:
    col = _collection(cfg)
    res = col.get(include=["metadatas", "documents"])  # no "ids" in include
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(len(res["ids"])):
            row = {
                "id": res["ids"][i],
                "document": res["documents"][i],
                "metadata": res["metadatas"][i],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_path
