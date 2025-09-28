from typing import Dict, Optional
import wikipediaapi
import time
import os
import json
from hashlib import sha1

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "..", "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(topic: str) -> str:
    key = sha1(topic.strip().lower().encode("utf-8")).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"wiki_{key}.json")

def wiki_lookup(topic: str, ttl_hours: int = 24) -> Optional[Dict]:
    path = _cache_path(topic)
    now = int(time.time())
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if now - int(obj.get("_cached_at", 0)) < ttl_hours * 3600:
                return obj
        except Exception:
            pass

    ua = os.environ.get("WIKI_USER_AGENT", "module2-rag/1.0 (contact: you@example.com)")
    wiki = wikipediaapi.Wikipedia(language="en", user_agent=ua)
    page = wiki.page(topic)
    if not page.exists():
        return None

    sections = []
    def collect(secs, depth=0, max_depth=2):
        if depth > max_depth: return
        for s in secs:
            text = s.text or ""
            if text.strip():
                sections.append({"title": s.title, "text": text})
            collect(s.sections, depth+1, max_depth)
    collect(page.sections)

    obj = {
        "source_type": "wiki",
        "title": page.title,
        "url": page.fullurl,
        "summary": page.summary,
        "sections": sections,
        "_cached_at": now,
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return obj
