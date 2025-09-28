#!/usr/bin/env python3
"""
Smoke tests for Module 2 adapters and blending.
- Web search (Tavily)
- Wikipedia lookup
- YouTube transcript (captions)
"""

import os
import yaml

CFG_PATH = os.path.join(os.path.dirname(__file__), "config", "module2.yaml")
with open(CFG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

from adapters.web.tavily_adapter import search_web_tavily
from adapters.wikipedia.wiki_adapter import wiki_lookup
from adapters.youtube.yt_adapter import yt_transcript
from orchestrator.blender import rrf_blend, cap_per_domain, enforce_min_chars
from assemble_context import build_blended_prompt

def run_web(query: str):
    webcfg = CFG.get("web", {})
    return search_web_tavily(
        query=query,
        api_key=os.environ.get(webcfg.get("api_key_env","TAVILY_API_KEY")),
        max_results=int(webcfg.get("max_results", 5)),
        recency_days=webcfg.get("recency_days"),
        timeout_seconds=int(webcfg.get("timeout_seconds", 15)),
        allow_domains=webcfg.get("allow_domains") or None,
        deny_domains=webcfg.get("deny_domains") or None,
    )

def test_web(query: str):
    print(f"\n[WEB] {query}")
    items = run_web(query)
    for i, it in enumerate(items, 1):
        print(f"{i}. {it['title']} — {it['url']}")
    snippets = enforce_min_chars(items, CFG["orchestration"]["blend"]["min_snippet_chars"])
    fused = rrf_blend([snippets], k=60, C=60)
    blended = cap_per_domain(fused, CFG["orchestration"]["blend"]["max_per_domain"])
    prompt = build_blended_prompt(query, blended[:CFG["orchestration"]["blend"]["max_snippets_total"]])
    print("\n=== PROMPT PREVIEW ===\n")
    print(prompt[:1200], "...\n")

def test_wiki(topic: str):
    print(f"\n[WIKI] {topic}")
    obj = wiki_lookup(topic, ttl_hours=CFG.get("wikipedia",{}).get("cache_ttl_hours",24))
    if not obj:
        print("No wiki result.")
        return
    print(f"Title: {obj['title']}\nURL: {obj['url']}\nSummary: {obj['summary'][:300]}...")

def test_yt(url: str):
    print(f"\n[YOUTUBE] {url}")
    yt_cfg = CFG.get("youtube", {})
    res = yt_transcript(
        url,
        prefer_captions=yt_cfg.get("prefer_captions", True),
        whisper_model=yt_cfg.get("whisper_model", "medium"),
        max_duration_minutes=yt_cfg.get("max_duration_minutes", 120),
        compute_type=yt_cfg.get("compute_type", "auto"),
        beam_size=yt_cfg.get("beam_size", 5),
        vad_filter=yt_cfg.get("vad_filter", True),
        language=yt_cfg.get("language", "en"),
        tmp_dir=yt_cfg.get("tmp_dir", "/tmp"),
        audio_format=yt_cfg.get("audio_format", "m4a"),
    )
    if not res:
        print("No transcript (nothing returned).")
        return
    if "error" in res:
        print("Transcript error:", res["error"])
        return
    print(f"Video ID: {res['video_id']}\nHas captions: {res['has_captions']}\nSource: {res['source']}")
    print(f"Transcript (first 400 chars):\n{res['transcript'][:400]}...")

def main():
    test_web("AI regulation updates this month")
    test_wiki("Transformer (machine learning)")

if __name__ == "__main__":
    main()
