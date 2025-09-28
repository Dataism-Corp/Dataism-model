import os
import yaml, chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict

# --- path-safe config loading ---
BASE_DIR = os.path.dirname(__file__)
CFG_PATH = os.path.join(BASE_DIR, "config.yaml")

def load_cfg():
    with open(CFG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    # Normalize any relative paths in config to absolute paths (so CWD doesn't matter)
    def _abs(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(BASE_DIR, p)

    try:
        paths = cfg.get("paths", {})
        for k in ("index_dir", "raw_dir", "processed_dir", "observability_dir"):
            if k in paths:
                paths[k] = _abs(paths[k])
        cfg["paths"] = paths
    except Exception:
        pass

    return cfg

_cfg = load_cfg()

# --- chroma ---
def chroma():
    client = chromadb.PersistentClient(
        path=_cfg["paths"]["index_dir"],
        settings=Settings()
    )
    return client.get_or_create_collection("kb_main")

# --- singletons ---
_embed_model = None
_ce_model = None

def get_embedder():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(
            _cfg["embedding"]["model_name"],
            device=_cfg["embedding"]["device"]
        )
    return _embed_model

def get_reranker():
    global _ce_model
    if _ce_model is None:
        _ce_model = CrossEncoder(
            _cfg["reranker"]["model_name"],
            device=_cfg["reranker"]["device"]
        )
    return _ce_model

def rerank(candidates: List[Dict], query: str, top_k_final: int) -> List[Dict]:
    if not candidates:
        return []
    ce = get_reranker()
    pairs = [[query, c["document"]] for c in candidates]
    scores = ce.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k_final]

def retrieve(query: str) -> List[Dict]:
    coll = chroma()
    embedder = get_embedder()

    q_emb = embedder.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False
    )
    res = coll.query(
        query_embeddings=q_emb,
        n_results=_cfg["retrieval"]["top_k_retrieve"],
        include=["documents", "metadatas", "distances"],
    )

    docs = []
    for i in range(len(res["ids"][0])):
        docs.append({
            "id": res["ids"][0][i],
            "document": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "distance": float(res["distances"][0][i]),
        })

    ranked = rerank(docs, query, _cfg["retrieval"]["top_k_final"])

    # Filter by min_rerank_score to avoid junk on chit-chat
    min_score = float(_cfg["retrieval"].get("min_rerank_score", 0.0))
    ranked = [c for c in ranked if c.get("rerank_score", 0.0) >= min_score]

    return ranked
