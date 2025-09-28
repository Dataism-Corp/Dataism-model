import os, json, yaml
from typing import List, Dict
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

def load_cfg():
    with open("config.yaml","r") as f:
        return yaml.safe_load(f)

def load_jsonl(path: str) -> List[Dict]:
    with open(path,"r",encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def run_index():
    cfg = load_cfg()
    processed_dir = cfg["paths"]["processed_dir"]
    index_dir = cfg["paths"]["index_dir"]
    os.makedirs(index_dir, exist_ok=True)

    client = chromadb.PersistentClient(path=index_dir, settings=Settings(allow_reset=True))
    coll = client.get_or_create_collection("kb_main", metadata={"hnsw:space": "cosine"})

    embed_model = SentenceTransformer(cfg["embedding"]["model_name"], device=cfg["embedding"]["device"])
    embed_batch = cfg["embedding"]["batch_size"]

    try:
        coll.delete(where={})
    except Exception:
        pass

    files = [os.path.join(processed_dir, f) for f in os.listdir(processed_dir) if f.endswith(".jsonl")]
    texts, ids, metadatas = [], [], []

    for fp in tqdm(files, desc="Collecting chunks"):
        rows = load_jsonl(fp)
        for r in rows:
            cid = f"{r['doc_id']}::{r['section']}"
            ids.append(cid)
            texts.append(r["text"])
            metadatas.append({
                "doc_id": r["doc_id"],
                "title": r["title"],
                "page": r["page"] if r["page"] is not None else -1,
                "section": r["section"],
                "source_path": r["source_path"]
            })

    for i in tqdm(range(0, len(texts), embed_batch), desc="Embedding & upserting"):
        batch_texts = texts[i:i+embed_batch]
        batch_ids = ids[i:i+embed_batch]
        batch_meta = metadatas[i:i+embed_batch]
        embs = embed_model.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False)
        coll.add(ids=batch_ids, embeddings=embs.tolist(), metadatas=batch_meta, documents=batch_texts)

    print(f"✅ Indexed {len(texts)} chunks into Chroma at {index_dir}")

if __name__ == "__main__":
    run_index()
