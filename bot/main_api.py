from fastapi import FastAPI, Request, Header, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio, json, os, sqlite3, datetime, re, yaml
from dotenv import load_dotenv
from typing import List, Dict
from tools import kill

# ✅ fixed imports
from tools import apis
from tools.media import read_url, read_pdf_bytes
import model_chat, model_code
from bots.telegram_bot import clean_response_prefixes

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

app = FastAPI()
EXPECTED_TOKEN = os.getenv("BEARER_TOKEN", "changeme")
SYSTEM_PROMPT = "Answer succinctly in natural language. Do not include system or reasoning steps."
DB_PATH = os.path.join(PROJECT_ROOT, "memory.db")

# Router config
ROUTER_CFG = yaml.safe_load(open(os.path.join(PROJECT_ROOT, "router.yaml"))) if os.path.exists(os.path.join(PROJECT_ROOT, "router.yaml")) else {}

# --- Memory ---
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions(
            session_id TEXT,
            ts TEXT,
            role TEXT,
            content TEXT
        )
    """)
    conn.close()
init_db()

def save_message(session_id, role, content):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        "INSERT INTO sessions VALUES (?, ?, ?, ?)",
        (session_id, datetime.datetime.utcnow().isoformat(), role, content)
    )
    conn.commit()
    conn.close()

def get_history(session_id):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    rows = conn.execute(
        "SELECT role, content FROM sessions WHERE session_id=? ORDER BY ts",
        (session_id,)
    ).fetchall()
    conn.close()
    return rows

def build_messages(session_id, user_prompt):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for role, content in get_history(session_id):
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_prompt})
    return messages

# --- Router ---
def route_is_code(text: str) -> bool:
    """Determine if a user query is code-related."""
    cfg = ROUTER_CFG or {}
    rules = (cfg.get("rules") or {}).get("code_regex", [])
    
    # Check custom regex patterns first
    for pat in rules:
        if re.search(pat, text, flags=re.IGNORECASE | re.DOTALL):
            return True
    
    # Explicit code keywords
    code_keywords = [
        "code", "function", "program", "script", "implement", "algorithm",
        "class", "method", "variable", "python", "javascript", "java", "c++",
        "typescript", "html", "css", "sql", "programming", "developer", "coding",
        "debug", "error", "compiler", "interpreter", "syntax", "api", "library",
        "framework", "module", "package", "dependency", "import", "export",
        "function", "method", "console", "print", "output", "return", "for loop", 
        "while loop", "if statement", "conditional", "array", "list", "dictionary",
        "object", "class", "instance", "parameter", "argument", "callback",
        "async", "await", "promise", "thread", "concurrent", "parallel"
    ]
    
    text_lower = text.lower()
    
    # Check for explicit code request indicators
    explicit_patterns = [
        r"write (?:a|some) code",
        r"write (?:a|an) (?:function|algorithm|program|script)",
        r"implement (?:a|an|the) (?:function|algorithm|class|method)",
        r"code (?:example|snippet)",
        r"fix (?:this|the|my) (?:code|bug|error)",
        r"debug (?:this|the|my) code"
    ]
    
    for pattern in explicit_patterns:
        if re.search(pattern, text_lower):
            return True
    
    # Check for keywords
    if any(keyword in text_lower for keyword in code_keywords):
        # If keywords are present, also check for code-like syntax
        if len(re.findall(r"[{}();=:]", text)) > 3:
            return True
        # Or check for common programming patterns
        if re.search(r"(def|function|var|let|const|class|if|for|while)\s+\w+", text):
            return True
        # Or check for a request to write/create/implement something
        if re.search(r"(write|create|implement|code|program)\s+", text_lower):
            return True
    
    # Original fallback syntax check - relaxed a bit
    return len(re.findall(r"[{}();=:]", text)) > 6

def guard_auth(authorization: str):
    if not authorization or authorization.split()[-1] != EXPECTED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")

# --- API: chat completions ---
@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: str = Header(None)):
    guard_auth(authorization)
    body = await request.json()
    session_id = body.get("session_id", "default")
    messages = body.get("messages", [])
    user_prompt = messages[-1]["content"] if messages else ""

    # Use the messages from payload if provided, otherwise build from history
    if messages:
        full_messages = messages
        # Still save the user message for conversation history
        save_message(session_id, "user", user_prompt)
    else:
        # Fallback to old behavior
        save_message(session_id, "user", user_prompt)
        full_messages = build_messages(session_id, user_prompt)

    is_code = route_is_code(user_prompt)
    stream_fn = model_code.stream_generate if is_code else model_chat.stream_generate

    async def generator():
        reply_accum = ""
        last_yielded_length = 0
        try:
            for token in stream_fn(full_messages):
                if kill.check_kill():
                    yield f"data: {json.dumps({'delta': '[⚠️ stopped by kill switch]'})}\n\n"
                    break

                reply_accum += token
                
                # Apply cleaning to accumulated response
                cleaned_accum = clean_response_prefixes(reply_accum)
                
                # Yield only the new cleaned content
                if len(cleaned_accum) > last_yielded_length:
                    new_content = cleaned_accum[last_yielded_length:]
                    if new_content:
                        yield f"data: {json.dumps({'delta': new_content})}\n\n"
                        last_yielded_length = len(cleaned_accum)
                
                await asyncio.sleep(0.005)
        finally:
            if reply_accum.strip():
                # Save the fully cleaned response
                cleaned_final = clean_response_prefixes(reply_accum)
                save_message(session_id, "assistant", cleaned_final.strip())
        yield "data: {\"done\":true}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


# --- Tools: health, config, reset ---
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/config")
def config():
    return {"router": ROUTER_CFG}

@app.post("/session/reset")
def reset_session():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    init_db()
    return {"status": "reset"}

@app.post("/tools/kill")
def trigger_kill():
    return kill.trigger_kill()

@app.post("/tools/kill/reset")
def reset_kill():
    return kill.reset_kill()

@app.get("/tools/kill/status")
def kill_status():
    return {"killed": kill.check_kill()}

# --- Tools: search ---
@app.get("/tools/search")
def api_search(q: str, topk: int = 3):
    return JSONResponse(apis.search_cse(q, topk=topk))

# --- Tools: read URL / PDF upload ---
@app.post("/tools/read_url")
def api_read_url(url: str = Form(...)):
    text = read_url(url)
    return {"text": text[:4000]}

@app.post("/tools/read_pdf")
async def api_read_pdf(file: UploadFile = File(...)):
    data = await file.read()
    text = read_pdf_bytes(data)

    if not text.strip():
        return {"summary": "❌ Could not extract text from PDF."}

    messages = [
        {"role": "system", "content": "You are Dateria. Summarize uploaded documents clearly and concisely."},
        {"role": "user", "content": f"Please summarize this document:\n\n{text[:8000]}"}  # cap length
    ]

    summary = "".join(model_chat.stream_generate(messages))
    return {"summary": summary.strip()}

# --- Desktop Control (dry-run example) ---
@app.post("/tools/desktop/plan")
def desktop_plan(task: str):
    plan = [
        {"step": "Focus window", "action": f"desktop.window.focus(title~='{task[:32]}*')"},
        {"step": "Type", "action": "desktop.type('hello')"},
        {"step": "Save", "action": "desktop.hotkey('ctrl+s') -> Desktop\\hello.txt"},
        {"step": "Screenshot (before/after)", "action": "desktop.screenshot('./logs/plan.png')"}
    ]
    return {"mode": "dry-run", "task": task, "plan": plan, "approval": "required"}

@app.post("/tools/desktop/execute")
def desktop_execute(task: str, approve: bool = False):
    if kill.check_kill():
        return {"status":"stopped","note":"killed by user"}
    if not approve:
        return {"status": "denied", "reason": "approval required",
                "hint": "POST with approve=true to proceed (Windows agent required)."}
    return {"status": "pending", "note": "Host Windows agent not configured. Please install WinAppDriver/pywinauto service."}
