# module3/core/writer.py
import time, hashlib
from .store import save_cards, search
from .schema import normalize_turn_to_cards

DEDUP_WINDOW_SEC = 24 * 3600  # not used strictly, but kept for future

def _canon_key(card):
    s = f"{card.get('type','')}|{card.get('subject','')}|{card.get('value','')}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _is_recent_dupe(card, now, cfg):
    key = _canon_key(card)
    recent = search(query=card.get('subject',''), cfg=cfg, top_k=50)
    for r in recent or []:
        md = r.get("metadata", {})
        k2 = _canon_key({"type": md.get("type",""),
                         "subject": md.get("subject",""),
                         "value": md.get("value","")})
        if k2 == key:
            return True
    return False

def write(turn, cfg):
    cards = normalize_turn_to_cards(turn, cfg)
    now = time.time()
    filtered = []
    for c in cards:
        if _is_recent_dupe(c, now, cfg):
            continue
        filtered.append(c)
    if not filtered:
        return []
    return save_cards(filtered, cfg)
