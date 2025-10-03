"""
decay.py — TTL/decay/merge/cleanup routines (skeleton).
Wire these to a periodic task or call on startup/shutdown.
"""

from typing import Dict
from . import store

def maintenance(cfg: Dict):
    """
    Placeholder: later implement TTL expiry, duplicate merge, and compaction.
    For now we just add a log entry.
    """
    store._log_event(cfg, "maintenance tick (no-op)")