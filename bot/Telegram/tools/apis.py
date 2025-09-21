# tools/apis.py
import os
import time
import json
import requests
import feedparser
from urllib.parse import quote_plus, quote
from rag_store import cache_get, cache_set, add_documents
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

# default user agent
UA = {"User-Agent": "Dateria/1.0 (+https://local)"}

# ----------------- helpers -----------------
def _req_json(url, headers=None, retries=2, timeout=12, backoff=0.6):
    last = None
    h = headers or UA
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=h, timeout=timeout)
            if r.ok:
                # some endpoints return text (atom) -> caller must handle
                try:
                    return r.json()
                except Exception:
                    return r.text
            last = r.text
        except Exception as e:
            last = str(e)
        if attempt < retries:
            time.sleep(backoff * (1 + attempt))
    raise RuntimeError(f"GET failed: {url} :: {last}")

def _cache_or_fetch(cache_key, ttl_seconds, fn):
    cached = cache_get(cache_key)
    if cached:
        return cached
    data = fn()
    try:
        cache_set(cache_key, data, ttl_seconds)
    except Exception:
        # non-fatal - still return data
        pass
    return data

def _normalize(items, source):
    out = []
    for it in items:
        title = (it.get("title") or "").strip()
        snippet = (it.get("snippet") or "").strip()
        link = (it.get("link") or "").strip()
        if title or snippet or link:
            out.append({"title": title, "snippet": snippet, "link": link, "source": source})
    return out

# small safe wrapper to add to RAG memory (non-blocking best-effort)
def _save_to_rag(items):
    try:
        if items:
            add_documents(items)
    except Exception:
        # don't fail the tool if RAG store is misbehaving
        pass

# ----------------- Google CSE (kept for compatibility) -----------------
def search_cse(q, topk=5):
    if not (GOOGLE_API_KEY and GOOGLE_CSE_ID):
        return [{"title":"Google CSE keys missing", "snippet":"Set GOOGLE_API_KEY and GOOGLE_CSE_ID in .env", "link":"", "source":"google-cse"}]
    url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}&q={quote_plus(q)}"
    key = f"cse::{q}::{topk}"
    def _fetch():
        j = _req_json(url)
        items = []
        for it in (j.get("items") or [])[:topk]:
            items.append({"title": it.get("title",""), "snippet": it.get("snippet",""), "link": it.get("link","")})
        out = _normalize(items, "google-cse")
        _save_to_rag(out)
        return out
    return _cache_or_fetch(key, 24*3600, _fetch)

# ----------------- 1) Wikipedia REST (search + summary) -----------------
def wiki(q: str, topk: int = 5):
    """
    Returns up to `topk` wiki search summary results (title, snippet, link).
    Uses:
      - Search: https://en.wikipedia.org/w/rest.php/v1/search/title?q={q}&limit={topk}
      - Summary: https://en.wikipedia.org/api/rest_v1/page/summary/{title_for_url}
    """
    key = f"wiki:{q}::{topk}"
    def _fetch():
        search_url = f"https://en.wikipedia.org/w/rest.php/v1/search/title?q={quote_plus(q)}&limit={topk}"
        j = _req_json(search_url)
        pages = j.get("pages") or []
        if not pages:
            return [{"title":"Not found", "snippet": f"No Wikipedia results for '{q}'.", "link":"", "source":"wikipedia"}]
        results = []
        for p in pages:
            title = p.get("title","").strip()
            # build canonical path: replace spaces with underscores then percent-encode
            title_for_url = title.replace(" ", "_")
            summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title_for_url)}"
            try:
                sumj = _req_json(summary_url)
                snippet = (sumj.get("extract") or "").strip()
                if not snippet:
                    snippet = p.get("excerpt","").strip()
                link = (sumj.get("content_urls",{}) or {}).get("desktop",{}).get("page") or f"https://en.wikipedia.org/wiki/{quote(title_for_url)}"
                results.append({"title": sumj.get("title", title), "snippet": snippet, "link": link})
            except Exception:
                # fallback to search excerpt + canonical link
                link = f"https://en.wikipedia.org/wiki/{quote(title_for_url)}"
                results.append({"title": title, "snippet": p.get("excerpt","").strip(), "link": link})
        out = _normalize(results, "wikipedia")
        _save_to_rag(out)
        return out
    return _cache_or_fetch(key, 24*3600, _fetch)

# ----------------- 2) Hacker News (Algolia) -----------------
def hn(q: str, topk: int = 5):
    key = f"hn::{q}::{topk}"
    def _fetch():
        url = f"https://hn.algolia.com/api/v1/search?query={quote_plus(q)}&hitsPerPage={topk}"
        j = _req_json(url)
        hits = j.get("hits") or []
        out_items = []
        for it in hits[:topk]:
            title = it.get("title") or it.get("story_title") or ""
            snippet = (it.get("story_text") or it.get("comment_text") or "")[:500]
            link = it.get("url") or f"https://news.ycombinator.com/item?id={it.get('objectID')}"
            out_items.append({"title": title, "snippet": snippet, "link": link})
        norm = _normalize(out_items, "hackernews")
        _save_to_rag(norm)
        return norm
    return _cache_or_fetch(key, 24*3600, _fetch)

# ----------------- 3) arXiv -----------------
def arxiv(q: str, topk: int = 5):
    key = f"arxiv::{q}::{topk}"
    def _fetch():
        url = f"http://export.arxiv.org/api/query?search_query=all:{quote_plus(q)}&max_results={topk}"
        feed = feedparser.parse(url)
        items = []
        for e in feed.entries[:topk]:
            title = e.get("title","")
            snippet = (e.get("summary","") or "").strip()
            link = e.get("link","")
            items.append({"title": title, "snippet": snippet, "link": link})
        norm = _normalize(items, "arxiv")
        _save_to_rag(norm)
        return norm
    return _cache_or_fetch(key, 24*3600, _fetch)

# ----------------- 4) OpenAlex -----------------
def openalex(q: str, topk: int = 5):
    key = f"openalex::{q}::{topk}"
    def _fetch():
        url = f"https://api.openalex.org/works?search={quote_plus(q)}&per-page={topk}"
        j = _req_json(url)
        items = []
        for it in (j.get("results") or [])[:topk]:
            title = it.get("title","")
            # try to build a readable snippet from abstract inverted index
            abstract_idx = it.get("abstract_inverted_index")
            snippet = ""
            if abstract_idx and isinstance(abstract_idx, dict):
                snippet = " ".join(list(abstract_idx.keys())[:50])
            link = it.get("id","")
            items.append({"title": title, "snippet": snippet, "link": link})
        norm = _normalize(items, "openalex")
        _save_to_rag(norm)
        return norm
    return _cache_or_fetch(key, 24*3600, _fetch)

# ----------------- 5) DexScreener (crypto pairs) -----------------
def dex(symbol: str, topk: int = 5):
    key = f"dex::{symbol}::{topk}"
    def _fetch():
        url = f"https://api.dexscreener.com/latest/dex/search/?q={quote_plus(symbol)}"
        j = _req_json(url)
        pairs = j.get("pairs") or []
        items = []
        for it in pairs[:topk]:
            title = f"{it.get('baseToken',{}).get('symbol','')} / {it.get('quoteToken',{}).get('symbol','')}"
            price = it.get("priceUsd","")
            change = it.get("priceChange",{}).get("h24","")
            snippet = f"Price: ${price} | 24h: {change}%"
            link = it.get("url","")
            items.append({"title": title, "snippet": snippet, "link": link})
        norm = _normalize(items, "dexscreener")
        _save_to_rag(norm)
        return norm
    # markets are fresher -> shorter cache
    return _cache_or_fetch(key, 6*3600, _fetch)

# ----------------- 6) CoinGecko (market data) -----------------
def coingecko(coin: str, topk: int = 1):
    key = f"cg::{coin}::{topk}"
    def _fetch():
        url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids={quote_plus(coin)}&per_page={topk}&page=1"
        j = _req_json(url)
        items = []
        for it in j[:topk]:
            title = it.get("name","")
            price = it.get("current_price","")
            change = it.get("price_change_percentage_24h","")
            link = f"https://www.coingecko.com/en/coins/{(it.get('id') or coin)}"
            snippet = f"Price: ${price} | 24h: {change}% | MC: {it.get('market_cap','')}"
            items.append({"title": title, "snippet": snippet, "link": link})
        norm = _normalize(items, "coingecko")
        _save_to_rag(norm)
        return norm
    return _cache_or_fetch(key, 6*3600, _fetch)

# ----------------- 7) RSS/Atom generic -----------------
def rss(url: str, topk: int = 5):
    key = f"rss::{url}::{topk}"
    def _fetch():
        feed = feedparser.parse(url)
        out = []
        for e in (feed.entries or [])[:topk]:
            title = e.get("title","")
            snippet = (e.get("summary") or e.get("description") or "")[:500]
            link = e.get("link","")
            out.append({"title": title, "snippet": snippet, "link": link})
        norm = _normalize(out, "rss")
        _save_to_rag(norm)
        return norm
    return _cache_or_fetch(key, 24*3600, _fetch)

# ----------------- 8) GitHub -----------------
def github(q: str, topk: int = 5):
    key = f"gh::{q}::{topk}"
    def _fetch():
        url = f"https://api.github.com/search/repositories?q={quote_plus(q)}&sort=stars&order=desc&per_page={topk}"
        j = _req_json(url)
        items = []
        for it in (j.get("items") or [])[:topk]:
            title = it.get("full_name","")
            snippet = (it.get("description","") or "")[:500]
            link = it.get("html_url","")
            items.append({"title": title, "snippet": snippet, "link": link})
        norm = _normalize(items, "github")
        _save_to_rag(norm)
        return norm
    return _cache_or_fetch(key, 24*3600, _fetch)

# ----------------- 9) StackOverflow (Stack Exchange) -----------------
def stackoverflow(q: str, topk: int = 5):
    key = f"so::{q}::{topk}"
    def _fetch():
        url = f"https://api.stackexchange.com/2.3/search?order=desc&sort=activity&intitle={quote_plus(q)}&site=stackoverflow&pagesize={topk}"
        j = _req_json(url)
        items = []
        for it in (j.get("items") or [])[:topk]:
            title = it.get("title","")
            snippet = f"score: {it.get('score','')}  | answers: {it.get('answer_count','')}"
            link = it.get("link","")
            items.append({"title": title, "snippet": snippet, "link": link})
        norm = _normalize(items, "stackoverflow")
        _save_to_rag(norm)
        return norm
    return _cache_or_fetch(key, 24*3600, _fetch)

# ----------------- Optional: YouTube (kept) -----------------
def youtube(q: str, topk: int = 5):
    if not YOUTUBE_API_KEY:
        return [{"title":"YouTube API key missing","snippet":"Set YOUTUBE_API_KEY in .env","link":"","source":"youtube"}]
    key = f"yt::{q}::{topk}"
    def _fetch():
        url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&type=video&maxResults={topk}&q={quote_plus(q)}&key={YOUTUBE_API_KEY}"
        j = _req_json(url)
        out = []
        for it in (j.get("items") or []):
            title = it["snippet"]["title"]
            desc  = it["snippet"].get("description","")
            vid   = it["id"]["videoId"]
            link  = f"https://www.youtube.com/watch?v={vid}"
            out.append({"title": title, "snippet": desc, "link": link})
        norm = _normalize(out, "youtube")
        _save_to_rag(norm)
        return norm
    return _cache_or_fetch(key, 24*3600, _fetch)
