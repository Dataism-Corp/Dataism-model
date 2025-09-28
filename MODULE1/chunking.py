from typing import List, Dict
import math

def _approx_tokens(s: str) -> int:
    return max(1, math.ceil(len(s) / 4))

def simple_chunk(text: str, target_tokens: int = 1000, overlap_tokens: int = 150) -> List[str]:
    if not text.strip():
        return []
    words = text.split()
    approx_tok_per_word = max(1, _approx_tokens(text)//max(1,len(words)))
    target_words = max(50, target_tokens // approx_tok_per_word)
    overlap_words = max(10, overlap_tokens // approx_tok_per_word)
    chunks = []
    i = 0
    while i < len(words):
        j = min(len(words), i + target_words)
        chunk = " ".join(words[i:j]).strip()
        if chunk:
            chunks.append(chunk)
        if j == len(words): break
        i = max(0, j - overlap_words)
    return chunks

def page_chunks(pages: List[Dict], target_tokens: int = 1000, overlap_tokens: int = 150) -> List[Dict]:
    all_chunks = []
    for p in pages:
        c_list = simple_chunk(p.get("text",""), target_tokens, overlap_tokens)
        for idx, c in enumerate(c_list):
            all_chunks.append({"page": p.get("page"), "chunk_index": idx, "text": c})
    return all_chunks
