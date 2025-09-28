import os, hashlib, json, time, yaml
from typing import Dict, Any, List
from text_extractors import extract_any
from chunking import simple_chunk, page_chunks

EXCLUDE_BASENAMES = {"README.md", ".DS_Store"}
ALLOWED_EXTS = {".pdf", ".docx", ".txt", ".md", ".markdown", ".json", ".jsonl", ".csv"}


def load_cfg():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def file_checksum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def meta_from_path(path: str) -> Dict[str, Any]:
    stat = os.stat(path)
    created_at = int(stat.st_mtime)
    return {
        "title": os.path.basename(path),
        "source_path": os.path.abspath(path),
        "created_at": created_at,
        "tags": []
    }


def save_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _text_len_from_extracted(extracted: Dict[str, Any]) -> int:
    if extracted["type"] == "pdf":
        return sum(len(p.get("text", "")) for p in extracted.get("pages", []))
    return len(extracted.get("text", ""))


def run_ingest():
    cfg = load_cfg()
    raw_dir = cfg["paths"]["raw_dir"]
    processed_dir = cfg["paths"]["processed_dir"]
    docstore_dir = cfg["paths"]["docstore_dir"]
    min_chars = int(cfg.get("chunking", {}).get("min_chars_for_index", 600))  # new knob

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(docstore_dir, exist_ok=True)

    index_manifest = []
    total_files = 0
    skipped_files = 0
    indexed_files = 0

    for root, _, files in os.walk(raw_dir):
        for name in files:
            total_files += 1
            if name in EXCLUDE_BASENAMES:
                skipped_files += 1
                print(f"[SKIP basename] {name}")
                continue

            src = os.path.join(root, name)
            ext = os.path.splitext(src)[1].lower()
            if ext not in ALLOWED_EXTS:
                skipped_files += 1
                print(f"[SKIP ext] {src}")
                continue

            print(f"[INGEST] {src}")
            meta = meta_from_path(src)
            checksum = file_checksum(src)
            doc_id = hashlib.sha1((meta["source_path"] + checksum).encode("utf-8")).hexdigest()[:16]
            meta["doc_id"] = doc_id
            meta["checksum"] = checksum

            try:
                extracted = extract_any(src)
            except Exception as e:
                print(f"[SKIP invalid] {src} ({e})")
                skipped_files += 1
                continue

            total_len = _text_len_from_extracted(extracted)
            if total_len < min_chars:
                skipped_files += 1
                print(f"[SKIP small] {src} (chars={total_len} < min={min_chars})")
                continue

            out_processed = os.path.join(processed_dir, f"{doc_id}.jsonl")
            out_docstore = os.path.join(docstore_dir, f"{doc_id}.json")

            records: List[Dict[str, Any]] = []
            if extracted["type"] == "pdf":
                chunks = page_chunks(
                    extracted["pages"],
                    cfg["chunking"]["target_tokens"],
                    cfg["chunking"]["overlap_tokens"]
                )
                for c in chunks:
                    records.append({
                        "doc_id": doc_id,
                        "title": meta["title"],
                        "page": c["page"],
                        "section": f"page_{c['page']}",
                        "text": c["text"],
                        "source_path": meta["source_path"],
                        "checksum": checksum,
                        "tags": meta["tags"]
                    })
            else:
                text = extracted.get("text", "")
                chunks = simple_chunk(
                    text,
                    cfg["chunking"]["target_tokens"],
                    cfg["chunking"]["overlap_tokens"]
                )
                for i, c in enumerate(chunks):
                    records.append({
                        "doc_id": doc_id,
                        "title": meta["title"],
                        "page": None,
                        "section": f"chunk_{i}",
                        "text": c,
                        "source_path": meta["source_path"],
                        "checksum": checksum,
                        "tags": meta["tags"]
                    })

            if not records:
                skipped_files += 1
                print(f"[SKIP empty] {src} (no chunks)")
                continue

            with open(out_processed, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            save_json(out_docstore, {
                "meta": meta,
                "preview": records[0]["text"][:1200]
            })

            indexed_files += 1
            index_manifest.append({
                "doc_id": doc_id,
                "title": meta["title"],
                "path": meta["source_path"],
                "records": len(records)
            })

    save_json(os.path.join(processed_dir, "_manifest.json"), {
        "generated_at": int(time.time()),
        "docs": index_manifest,
        "stats": {
            "total_files_seen": total_files,
            "indexed_files": indexed_files,
            "skipped_files": skipped_files
        }
    })
    print(f"✅ Ingestion complete. Indexed={indexed_files} Skipped={skipped_files} Seen={total_files}")


if __name__ == "__main__":
    run_ingest()
