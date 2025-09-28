import os
import yaml
import re
from typing import Tuple

CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "module2.yaml")

_GREET_PAT = re.compile(r"^(hi|hello|hey|yo|sup|thanks|thank you|ok|okay)\b[.!?]*$", re.I)
_QWORDS = {"what","how","why","when","where","which","who","does","do","is","are","can","should"}

def load_cfg():
    with open(CFG_PATH, "r") as f:
        return yaml.safe_load(f)

def mentions_recency(q: str) -> bool:
    ql = q.lower()
    return any(w in ql for w in ["today", "this week", "this month", "latest", "breaking", "2024", "2025"])

def should_use_web(question: str, kb_confident: bool, explicit_request: bool) -> Tuple[bool, dict]:
    cfg = load_cfg()
    pol = cfg.get("orchestration", {}).get("use_web_when", {})
    if explicit_request and pol.get("explicit_web_request", True):
        return True, {"reason": "explicit"}
    if _GREET_PAT.match(question.strip()):
        return False, {"reason": "greeting"}
    if pol.get("recency_mentioned", True) and mentions_recency(question):
        return True, {"reason": "recency"}
    if pol.get("kb_low_confidence", True) and not kb_confident:
        return True, {"reason": "kb_low_conf"}
    return False, {"reason": "kb_sufficient"}
