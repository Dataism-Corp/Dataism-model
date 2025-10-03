# Module 3 — Knowledge Memory

This module adds short-term and long-term memory so the agent can remember preferences,
decisions, facts, and tasks across sessions.

## Components
- **config/module3.yaml** — knobs for recall, decay, TTLs, paths, and controls.
- **core/schema.py** — memory card data model.
- **core/writer.py** — captures candidates after each turn and stores them.
- **core/retriever.py** — recalls relevant memories (semantic + recency).
- **core/store.py** — persistence (vector DB + JSON backups).
- **core/decay.py** — TTL/decay/merge maintenance.
- **core/controls.py** — `/memory` commands.

## Next steps
1. Wire `store.py` to your embedder + Chroma (new collection: `memories`).
2. Call `retriever.recall(...)` before RAG/Web, inject as `[MEMORY CONTEXT]`.
3. Call `writer.write(turn, cfg)` after each assistant reply.
4. Expose `/memory list|export|purge|forget|off|on` via your chat loop.