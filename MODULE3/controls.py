# module3/core/controls.py
import os, time
from .store import search, export_jsonl

def _dedup(cards):
    seen, out = set(), []
    for c in cards or []:
        key = (c.get("metadata", {}).get("type",""),
               c.get("metadata", {}).get("subject",""),
               c.get("metadata", {}).get("value",""))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out

def _backups_dir(cfg):
    # Try config path first; else default to module3/data/backups
    p = (cfg.get("paths", {}) or {}).get("backups_dir")
    if not p:
        here = os.path.dirname(os.path.dirname(__file__))  # module3/core -> module3
        p = os.path.join(here, "data", "backups")
    os.makedirs(p, exist_ok=True)
    return p

def handle(cmd: str, cfg):
    cmd = " ".join((cmd or "").strip().split()).lower()

    if cmd == "/memory list":
        cards = _dedup(search(query="", cfg=cfg, top_k=100))
        if not cards:
            return "No memories yet."
        lines = ["Recent memories:"]
        for h in cards[:50]:
            md = h.get("metadata", {})
            lines.append(f"- [{md.get('type','')}] {md.get('subject','')}: {md.get('value','')}")
        return "\n".join(lines)

    if cmd == "/memory export":
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join(_backups_dir(cfg), f"memories-{ts}.jsonl")
        path = export_jsonl(out_path, cfg)
        return f"Exported memories to: {path}"

    return "Unknown memory command."
