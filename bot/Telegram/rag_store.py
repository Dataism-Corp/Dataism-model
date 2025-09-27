import os, sqlite3, json, time, hashlib
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer

# Use the same DB path as the main project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, "memory.db")
EMB_MODEL_ID = "BAAI/bge-large-en-v1.5"
_model = None

# --------- Embedding Model ---------
def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(
            EMB_MODEL_ID,
            device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"
        )
    return _model


def init_db():
    """Initialize database with retry logic for locked database"""
    import time
    
    for attempt in range(3):
        try:
            conn = sqlite3.connect(DB_PATH, timeout=10.0)
            c = conn.cursor()

            # memory table
            c.execute("""CREATE TABLE IF NOT EXISTS memory(
                id TEXT PRIMARY KEY,
                ts TEXT,
                session_id TEXT,
                source TEXT,
                title TEXT,
                snippet TEXT,
                url TEXT,
                content TEXT,
                emb BLOB
            )""")

            # cache table
            c.execute("""CREATE TABLE IF NOT EXISTS cache(
                key TEXT PRIMARY KEY,
                ts INTEGER,
                ttl INTEGER,
                payload TEXT
            )""")

            # sessions table for chat logs
            c.execute("""CREATE TABLE IF NOT EXISTS sessions(
                session_id TEXT,
                ts TEXT,
                role TEXT,
                content TEXT
            )""")

            # routing_logs table for analytics
            c.execute("""CREATE TABLE IF NOT EXISTS routing_logs(
                ts TEXT,
                user_text TEXT,
                selected_tool TEXT,
                confidence REAL,
                reason TEXT
            )""")

            conn.commit()
            conn.close()
            print(f"✅ Database initialized successfully at {DB_PATH}")
            return
            
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and attempt < 2:
                print(f"⚠️ Database locked, retrying in {attempt + 1} seconds...")
                time.sleep(attempt + 1)
                continue
            else:
                print(f"⚠️ Database initialization warning: {e}")
                # Continue anyway, tables might exist
                return
        except Exception as e:
            print(f"⚠️ Database initialization error: {e}")
            return


init_db()

def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# --------- Cache ---------
def cache_get(key: str):
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        c = conn.cursor()
        
        # Ensure cache table exists
        c.execute("""CREATE TABLE IF NOT EXISTS cache(
            key TEXT PRIMARY KEY,
            ts INTEGER,
            ttl INTEGER,
            payload TEXT
        )""")
        
        c.execute("SELECT ts, ttl, payload FROM cache WHERE key=?", (key,))
        row = c.fetchone()
        conn.close()
        
        if not row:
            return None
        ts, ttl, payload = row
        if int(time.time()) - ts > ttl:
            return None
        try:
            return json.loads(payload)
        except Exception:
            return None
            
    except sqlite3.OperationalError as e:
        print(f"⚠️ Cache get error: {e}")
        return None
    except Exception as e:
        print(f"⚠️ Cache get error: {e}")
        return None

def cache_set(key: str, payload: dict, ttl_sec: int = 86400):
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        c = conn.cursor()
        
        # Ensure cache table exists
        c.execute("""CREATE TABLE IF NOT EXISTS cache(
            key TEXT PRIMARY KEY,
            ts INTEGER,
            ttl INTEGER,
            payload TEXT
        )""")
        
        c.execute("REPLACE INTO cache(key, ts, ttl, payload) VALUES(?,?,?,?)",
                  (key, int(time.time()), ttl_sec, json.dumps(payload)[:2_000_000]))
        conn.commit()
        conn.close()
        
    except sqlite3.OperationalError as e:
        print(f"⚠️ Cache set error: {e}")
    except Exception as e:
        print(f"⚠️ Cache set error: {e}")

# --------- Embeddings ---------
def _embed(texts):
    model = _get_model()
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vecs.astype(np.float32)

# --------- Add Docs ---------
def add_documents(items, session_id: str = "default"):
    """
    items: list of dicts with keys: title, snippet, link, source
    """
    if not items:
        return 0
    docs = []
    for it in items:
        title = it.get("title", "").strip()
        snippet = it.get("snippet", "").strip()
        link = it.get("link", "").strip()
        source = it.get("source", "").strip()
        content = (title + "\n" + snippet + "\n" + link).strip()
        if content:
            docs.append((title, snippet, link, source, content))
    if not docs:
        return 0

    embeddings = _embed([d[4] for d in docs])

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    ts = datetime.utcnow().isoformat()
    for (title, snippet, link, source, content), emb in zip(docs, embeddings):
        doc_id = _hash(content + ts + source + session_id)
        c.execute("""REPLACE INTO memory
                     (id, ts, session_id, source, title, snippet, url, content, emb)
                     VALUES(?,?,?,?,?,?,?,?,?)""",
                  (doc_id, ts, session_id, source, title, snippet, link, content, emb.tobytes()))
    conn.commit()
    conn.close()
    return len(docs)

# --------- Search Docs ---------
def search_documents(query: str, session_id: str = "default", topk: int = 5):
    """
    Semantic search scoped to a session_id.
    Returns: list of dicts with title, snippet, link, source, score
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, title, snippet, url, source, content, emb FROM memory WHERE session_id=?", (session_id,))
    rows = c.fetchall()
    conn.close()
    if not rows:
        return []

    # Encode query
    q_emb = _embed([query])[0]

    docs, embeddings = [], []
    for (doc_id, title, snippet, url, source, content, emb_bytes) in rows:
        try:
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            embeddings.append(emb)
            docs.append({
                "id": doc_id,
                "title": title,
                "snippet": snippet,
                "link": url,
                "source": source,
                "content": content,
            })
        except Exception:
            continue

    if not embeddings:
        return []

    embeddings = np.vstack(embeddings)  # (N, D)

    # cosine similarity
    scores = np.dot(embeddings, q_emb)
    top_idx = np.argsort(scores)[::-1][:topk]

    results = []
    for i in top_idx:
        d = docs[i]
        results.append({
            "title": d["title"],
            "snippet": d["snippet"],
            "link": d["link"],
            "source": d["source"],
            "score": float(scores[i]),
        })
    return results
