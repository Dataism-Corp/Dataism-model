# module3/core/schema.py
import re
from typing import Dict, List
from datetime import datetime

def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _card(_type: str, subject: str, value: str, evidence: str) -> Dict:
    return {
        "type": _type,               # 'preference' | 'fact' | 'note'
        "subject": subject.strip(),
        "value": value.strip(),
        "evidence": evidence,
        "timestamp": now_iso(),
    }

# --- compiled patterns ---
_BULLET_WORDS = r"(bullet(?:s)?|bulleted|bullet\s*points?)"
_STYLE_VERBS  = r"(answer|respond|reply|format|write|present|give)"
_NOW_PHRASE   = r"(from\s+now\s+on|going\s+forward|always|by\s+default)"
_ONLY_OPT     = r"(only\s+)?"

_BULLET_PREF_PAT = re.compile(
    rf"\b{_STYLE_VERBS}\b.*?\b{_ONLY_OPT}{_BULLET_WORDS}\b|\b{_BULLET_WORDS}\b.*?\b{_STYLE_VERBS}\b",
    flags=re.I | re.S,
)
_BULLET_STRONG_PREF_PAT = re.compile(
    rf"{_NOW_PHRASE}.*?\b{_BULLET_WORDS}\b|\b{_BULLET_WORDS}\b.*?{_NOW_PHRASE}",
    flags=re.I | re.S,
)

def normalize_turn_to_cards(turn: Dict, cfg: Dict) -> List[Dict]:
    """
    Turn -> list of memory 'cards'.
    Rules are intentionally simple & conservative.
    """
    cards: List[Dict] = []
    user = (turn.get("user") or "").strip()
    assistant = (turn.get("assistant") or "").strip()

    # --- Preference: bullets (user asks for bullets) ---
    if _BULLET_PREF_PAT.search(user) or _BULLET_STRONG_PREF_PAT.search(user):
        cards.append(_card(
            "preference",
            subject="answer_style",
            value="bullets",
            evidence="user turn",
        ))

    # --- Preference: bullets (assistant answered in bullets) ---
    if re.search(r"^\s*[-•]\s", assistant, flags=re.M):
        cards.append(_card(
            "preference",
            subject="answer_style",
            value="bullets",
            evidence="assistant turn",
        ))

    # --- Fact: project codename ---
    m = re.search(r"\bcodename\s+is\s+([A-Za-z0-9_-]+)", user, flags=re.I)
    if m:
        cards.append(_card("fact", "project_codename", m.group(1), "user turn"))

    # --- Fact: kickoff date ---
    mk = re.search(r"\b(kickoff|kick-?off)\s+is\s+(on\s+)?([A-Za-z0-9 ,/-]+)", user, flags=re.I)
    if mk:
        cards.append(_card("fact", "kickoff_date", mk.group(3).strip(), "user turn"))

    return cards
