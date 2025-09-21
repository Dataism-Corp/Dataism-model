import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import json, requests, telebot, datetime, sqlite3
from dotenv import load_dotenv
from tools import apis
from tools.desktop import execute, policy, approvals
from tools.desktop.execute import run_desktop_command, WIN_IP
from tools import kill, uploads, media
import re
from rag_store import add_documents, search_documents, init_db
from tools.media import read_url
import model_chat

# Import verification system
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from verification_bot import handle_verify_command
    VERIFICATION_AVAILABLE = True
except ImportError:
    print("⚠️ Verification system not available")
    VERIFICATION_AVAILABLE = False

# Ensure database is initialized
init_db()

# Set timezone (adjust as needed)
os.environ['TZ'] = 'America/New_York'  # or 'UTC' for UTC
try:
    import pytz
    TIMEZONE = pytz.timezone('America/New_York')  # or pytz.UTC
except ImportError:
    TIMEZONE = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
BEARER = os.getenv("BEARER_TOKEN")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")
API = "http://127.0.0.1:8080" 
DB_PATH = os.path.join(PROJECT_ROOT, "memory.db")

# Load authorized desktop chat IDs
AUTHORIZED_DESKTOP_CHAT_IDS = os.getenv("AUTHORIZED_DESKTOP_CHAT_IDS", "").split(",")
AUTHORIZED_DESKTOP_CHAT_IDS = [cid.strip() for cid in AUTHORIZED_DESKTOP_CHAT_IDS if cid.strip()]

if not TOKEN or not BEARER:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or BEARER_TOKEN in .env")

bot = telebot.TeleBot(TOKEN)
HEADERS = {"Authorization": f"Bearer {BEARER}", "Content-Type": "application/json"}


# ---------------- Utils ----------------
def _condense_verification_output(text: str) -> str:
    """Return only the final verification summary, dropping noisy test logs.
    Find the last line containing 'Verification' and keep from there to the end.
    """
    try:
        if not text:
            return text
        lines = text.splitlines()
        for i in range(len(lines)-1, -1, -1):
            if "Verification" in lines[i]:
                return "\n".join(lines[i:]).strip()
        return text.strip()
    except Exception:
        return text
def _get_last_document_context(session_key: str) -> str:
    """Fetch the latest stored document/image text for the given session.
    Falls back gracefully if schema uses 'id' instead of 'session_id'.
    """
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT content FROM sessions
                WHERE session_id = ? AND role IN ('document','image','photo','file')
                ORDER BY ts DESC LIMIT 1
                """,
                (session_key,),
            )
        except sqlite3.OperationalError:
            # Fallback to 'id' column name
            cur.execute(
                """
                SELECT content FROM sessions
                WHERE id = ? AND role IN ('document','image','photo','file')
                ORDER BY ts DESC LIMIT 1
                """,
                (session_key,),
            )
        row = cur.fetchone()
        conn.close()
        return row[0] if row and row[0] else ""
    except Exception as e:
        print(f"⚠️ get_last_document_context error: {e}")
        return ""
def _extract_query(text: str, cmd: str) -> str:
    """Extract query part from either '/cmd arg' or 'cmd arg' or return text when neither prefix used.
    Returns empty string if only the command without args was provided.
    Enhanced to handle natural language variations.
    """
    if not text:
        return ""
    t = text.strip()
    tl = t.lower()
    slash_prefix = f"/{cmd.lower()} "
    bare_prefix = f"{cmd.lower()} "
    
    # Handle slash commands
    if tl.startswith(slash_prefix):
        return t[len(slash_prefix):].strip()
    if tl == f"/{cmd.lower()}":
        return ""
    
    # Handle bare commands at start
    if tl.startswith(bare_prefix):
        return t[len(bare_prefix):].strip()
    if tl == cmd.lower():
        return ""
    
    # For natural language, if the command word appears anywhere, extract everything after it
    cmd_index = tl.find(cmd.lower())
    if cmd_index >= 0:
        after_cmd = t[cmd_index + len(cmd):].strip()
        return after_cmd
    
    # Fallback to returning the whole text (for cases where command is implied)
    return t

def safe_send(chat_id, text):
    # For now, just do basic cleaning and encoding
    clean = (text or "").encode("utf-8", "ignore").decode("utf-8")
    # Remove some obvious prefixes manually
    for prefix in [
        "ASSISTANT:", "SYSTEM:", "USER:",
        "Assistant:", "System:", "User:",
        "Dateria:", "dateria:"
    ]:
        if clean.startswith(prefix):
            clean = clean[len(prefix):].lstrip()
    
    for i in range(0, len(clean), 7600):
        bot.send_message(chat_id, clean[i:i+7600])


def format_items(items):
    lines = []
    for it in items:
        title = it.get("title", "").strip()
        snippet = it.get("snippet", "").strip()
        link = it.get("link", "").strip()
        src = it.get("source", "")
        line = f"• {title}\n{snippet}\n{link}\n(source: {src})"
        lines.append(line.strip())
    return "\n\n".join(lines) if lines else "No results."


def save_system_event(content: str):
    """Log kill/reset events into memory.db with role=system"""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions(id TEXT, ts TEXT, role TEXT, content TEXT)"
        )
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?)",
            ("global", datetime.datetime.utcnow().isoformat(), "system", content),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print("⚠️ DB log error:", e)
        conn.close()
    except Exception as e:
        print("⚠️ DB log error:", e)

# --- call_chat ---
def call_chat(msg, sid, code=False, chat_id=None, bot=None, stream=True, autosend=True, max_tokens=500, single_message=True):
    """
    Call the LLM API with optional RAG context.
    """
    # Decide if we should send a single Telegram message at the end (default True for all)
    # single_message can still be overridden per-call if needed

    # --- 🔹 NEW: Retrieve context from RAG ---
    # Always try to use RAG context for uploaded content, but be smart about when to include it
    use_rag = True
    simple_greeting_indicators = [
        "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
        "how are you", "what's up", "thanks", "thank you", "bye", "goodbye"
    ]
    
    # Skip RAG only for very simple greetings that clearly don't need context
    msg_lower = msg.lower().strip()
    is_simple_greeting = any(indicator in msg_lower for indicator in simple_greeting_indicators) and len(msg.split()) <= 3

    # For very short greetings, avoid calling the model and reply briefly
    if is_simple_greeting:
        use_rag = False
        reply = "Hi! How can I help you today?"
        if autosend and bot is not None and chat_id is not None:
            safe_send(chat_id, reply)
        return reply
    
    rag_context = None
    if use_rag:
        rag_context = uploads.retrieve_context(msg, session_id=str(sid), topk=3)
        # Clean the RAG context to remove conversation artifacts
        if rag_context:
            # Only clean if it contains conversation prefixes (not normal RAG content)
            if 'USER:' in rag_context or 'ASSISTANT:' in rag_context:
                rag_context = clean_response_prefixes(rag_context)
                # Additional cleaning for RAG context - be more careful
                rag_context = re.sub(r'USER:.*?(?=USER:|ASSISTANT:|$)', '', rag_context, flags=re.DOTALL | re.IGNORECASE)
                rag_context = re.sub(r'ASSISTANT:.*?(?=USER:|ASSISTANT:|$)', '', rag_context, flags=re.DOTALL | re.IGNORECASE)
                rag_context = rag_context.strip()

    # Determine if RAG context is actually relevant to the question
    use_context_in_response = False
    # Pre-compute if the user is explicitly asking about the uploaded artifact
    question_lower = msg.lower().strip()
    explicit_doc_indicators = ['pdf', 'document', 'doc', 'file', 'uploaded', 'text', 'image', 'photo', 'picture', 'screenshot', 'page', 'scan', 'ocr', 'figure', 'diagram', 'table', 'chart']
    explicit_doc_ref = any(ind in question_lower for ind in explicit_doc_indicators)

    # If user explicitly refers to the uploaded artifact but semantic search found nothing,
    # fall back to the most recent stored document for this session
    if explicit_doc_ref and (not rag_context or not rag_context.strip()):
        fallback_ctx = _get_last_document_context(str(sid))
        if fallback_ctx:
            rag_context = fallback_ctx

    if rag_context and len(rag_context.strip()) > 100:  # Higher minimum meaningful context length
        # Much more strict relevance check - only use context if there's clear topical overlap
        context_lower = rag_context.lower()
        
        # Extract key entities from the question (avoid common words completely)
        question_words = set()
        for word in question_lower.split():
            word = word.strip('.,!?;:\'"')
            # Only include meaningful names, places, or specific terms (length > 4 and not common words)
            if (len(word) > 4 and word not in [
                'what', 'when', 'where', 'which', 'that', 'this', 'with', 'from', 'have', 'been', 
                'were', 'does', 'will', 'would', 'could', 'should', 'about', 'their', 'there', 
                'these', 'those', 'donald', 'trump', 'biden', 'obama', 'wife', 'husband'
            ]):
                question_words.add(word)
        
        # Check for exact name matches or very specific terms only
        meaningful_overlap = 0
        for q_word in question_words:
            if q_word in context_lower and len(q_word) > 4:
                # Only count if it's a substantial match (not common English words)
                if context_lower.count(q_word) > 0 and q_word not in ['information', 'university', 'student', 'people', 'person']:
                    meaningful_overlap += 1
        
        # Calculate overlap ratio - now much more strict
        overlap_ratio = meaningful_overlap / max(len(question_words), 1)
        
        # Only use context if there's very substantial overlap (50%+) AND the question seems document-specific
        # For explicit references to the uploaded artifact (image/document/file/etc.), always allow context
        document_indicators = ['pdf', 'document', 'doc', 'file', 'uploaded', 'text', 'image', 'photo', 'picture', 'screenshot', 'page', 'scan', 'ocr', 'figure', 'diagram', 'table', 'chart', 'scholarship', 'university', 'faculty']
        has_document_context = any(indicator in question_lower for indicator in document_indicators)
        
        # Criteria: either high topical overlap OR explicit document reference
        # If the user explicitly refers to the uploaded artifact, use the context even without overlap
        if has_document_context:
            use_context_in_response = True
        elif (overlap_ratio > 0.5 and meaningful_overlap >= 2) or (has_document_context and overlap_ratio > 0.2):
            use_context_in_response = True

    if use_context_in_response:
        system_content = "You are a helpful assistant with access to uploaded documents. Answer questions directly and concisely using the provided context. Do not generate code unless specifically asked. Keep responses clear and to the point."
        user_content = f"""Context from uploaded files:
{rag_context}

Question: {msg}

Please answer based on the context above."""
    else:
        system_content = "You are a helpful assistant. Answer directly and concisely without any role prefixes. Provide clear, natural responses. Do not generate code unless specifically asked."
        user_content = ("```\n" + msg + "\n```") if code else msg

    payload = {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        "session_id": str(sid),
        "stream": stream,
        "max_tokens": max_tokens,
    }
    

    try:
        r = requests.post(
            f"{API}/v1/chat/completions",
            headers=HEADERS,
            json=payload,
            stream=stream,
            timeout=90,
        )
        r.raise_for_status()  # Raise an exception for bad status codes
    except Exception as e:
        print(f"⚠️ API request error: {e}")
        return f"❌ API error: {e}"

    # If not streaming, handle as single response
    if not stream:
        try:
            response_data = r.json()
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0].get("message", {}).get("content", "")
                return clean_response_prefixes(content)
            else:
                return "❌ No response from API"
        except Exception as e:
            print(f"⚠️ Non-streaming response error: {e}")
            return f"❌ Response parsing error: {e}"

    buffer, out = "", ""
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.strip():
            continue
        if line.strip() == "[DONE]":
            break
        if not line.startswith("data:"):
            continue
        try:
            data = json.loads(line[len("data: "):])
        except Exception as parse_error:
            print(f"⚠️ JSON parse error: {parse_error}, line: {line}")
            continue

        # Handle both 'delta' and 'content' fields, and check nested structures
        delta = ""
        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if "delta" in choice and "content" in choice["delta"]:
                delta = choice["delta"]["content"]
            elif "message" in choice and "content" in choice["message"]:
                delta = choice["message"]["content"]
        else:
            # Fallback to direct fields
            delta = data.get("delta", "") or data.get("content", "")
        
        if not delta:
            continue
            
        # Skip deltas that contain system/unwanted prefixes
        if any(prefix in delta for prefix in ["SYSTEM:", "USER:", "ASSISTANT:"]):
            print(f"⚠️ Skipping contaminated delta: {delta[:50]}...")
            continue

        out += delta
        buffer += delta

        # Apply cleaning periodically but not too frequently
        if len(buffer) >= 200:
            clean_buffer = clean_response_prefixes(buffer)
            buffer = clean_buffer

        if (not single_message) and autosend and bot and chat_id and len(buffer) >= 400:
            clean_buffer = clean_response_prefixes(buffer)
            if clean_buffer.strip():  # Only send if there's actual content
                bot.send_message(chat_id, clean_buffer)
            buffer = ""

    if (not single_message) and autosend and bot and chat_id and buffer:
        clean_buffer = clean_response_prefixes(buffer)
        if clean_buffer.strip():  # Only send if there's actual content
            bot.send_message(chat_id, clean_buffer)

    # Final cleaning of the complete response
    clean_out = clean_response_prefixes(out.strip())
    # Simple post-fixes for frequent spacing/typo artifacts
    clean_out = clean_out.replace("Value Error", "ValueError").replace("Type Error", "TypeError")
    clean_out = clean_out.replace("-- ", "— ").replace(" --", " —")
    clean_out = re.sub(r"\bnon\-\-\s*negative\b", "non-negative", clean_out, flags=re.IGNORECASE)
    # If single_message mode is on, send the entire response once
    if single_message and autosend and bot and chat_id:
        safe_send(chat_id, clean_out)
    return clean_out


def clean_response_prefixes(text):
    """Remove common chat prefixes and fix formatting issues in AI responses"""
    if not text:
        return text
    
    import re
    
    # First, fix any obvious encoding/spacing issues
    cleaned = text.strip()
    
    # List of prefixes to remove (case insensitive)
    prefixes_to_remove = [
        "ASSISTANT:", "Assistant:", "assistant:",
        "SYSTEM:", "System:", "system:",
        "USER:", "User:", "user:",
        "AI:", "ai:", "Ai:",
        "BOT:", "Bot:", "bot:",
        "RESPONSE:", "Response:", "response:",
        "Dateria:", "dateria:",
        "REPLY:", "Reply:", "reply:"
    ]
    
    # Remove prefixes
    for prefix in prefixes_to_remove:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break
    
    # Fix common code formatting issues
    # 1. Add spaces after punctuation when missing
    cleaned = re.sub(r'([.,:;])([a-zA-Z])', r'\1 \2', cleaned)
    
    # 2. Add spaces around operators when missing  
    cleaned = re.sub(r'([a-zA-Z0-9])([=+\-*/])([a-zA-Z0-9])', r'\1 \2 \3', cleaned)
    
    # 3. Add spaces after parentheses when missing
    cleaned = re.sub(r'(\))([a-zA-Z])', r'\1 \2', cleaned)
    
    # 4. Fix missing spaces after keywords
    cleaned = re.sub(r'(if|for|while|return|def|class)([a-zA-Z])', r'\1 \2', cleaned)
    
    # 5. Fix camelCase spacing (like TheCapitalOfIndia -> The Capital Of India)
    cleaned = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned)
    
    # 6. Add spaces after commas when missing
    cleaned = re.sub(r'([,])([a-zA-Z0-9])', r'\1 \2', cleaned)
    
    return cleaned


# ---------------- Utils ----------------
def safe_send(chat_id, text):
    """Send message with error handling and length checking."""
    try:
        if len(text) > 4096:
            # Split long messages
            for i in range(0, len(text), 4000):
                chunk = text[i:i+4000]
                bot.send_message(chat_id, chunk)
        else:
            bot.send_message(chat_id, text)
    except Exception as e:
        print(f"Error sending message: {e}")
        try:
            bot.send_message(chat_id, f"⚠️ Error: {str(e)[:100]}...")
        except:
            pass


# ---------- Summarization Helper ----------
def summarize_content(raw_text, user_id, chat_id):
    if not raw_text.strip():
        return safe_send(chat_id, "❌ No content to summarize.")

    # Store content in memory for follow-up Q&A
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions(session_id TEXT, ts TEXT, role TEXT, content TEXT)"
        )
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?)",
            (f"user-{user_id}", datetime.datetime.utcnow().isoformat(), "document", raw_text),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print("⚠️ DB log error (save doc):", e)

    summary_request = f"Summarize the following content in 5-8 concise sentences:\n\n{raw_text}"
    summary = call_chat(summary_request, user_id, code=False, chat_id=chat_id, bot=bot, autosend=False, max_tokens=800, single_message=True)
    safe_send(chat_id, f"📌 Summary:\n{summary}\n\n👉 Now ask me questions about this content directly.")


# -------- Smart Router with Confidence Scoring --------
def log_routing_decision(user_text: str, selected_tool: str, confidence: float, reason: str):
    """Log routing decisions for telemetry"""
    try:
        timestamp = datetime.datetime.utcnow().isoformat()
        log_entry = f"[{timestamp}] ROUTER: '{user_text[:50]}...' -> {selected_tool} (conf:{confidence:.2f}) - {reason}"
        print(log_entry)
        
        # Also save to database for analytics
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS routing_logs(ts TEXT, user_text TEXT, selected_tool TEXT, confidence REAL, reason TEXT)"
        )
        conn.execute(
            "INSERT INTO routing_logs VALUES (?, ?, ?, ?, ?)",
            (timestamp, user_text[:200], selected_tool, confidence, reason)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Routing log error: {e}")

def detect_explicit_command(user_text: str):
    """Check for explicit slash commands"""
    text = (user_text or "").strip()
    if text.startswith('/'):
        command = text.split()[0][1:].lower()  # Remove / and get first word
        valid_commands = ['wiki', 'search', 'read', 'desktop', 'yt', 'gh', 'so', 'cg', 'rss', 'reset', 'status', 'chat', 'code', 'time', 'approvals', 'desktop_status', 'approve', 'deny']
        if command in valid_commands:
            return command
    return None

def is_attached_file(user_text: str, msg_obj=None):
    """Detect if message has attached file or PDF URL"""
    text = (user_text or "").lower()
    
    # Check for PDF URLs
    if text.startswith("http") and ".pdf" in text:
        return True
        
    # Check for file attachments (would need message object)
    # This is handled separately in handle_upload()
    return False

def llm_tool_selector(user_text: str):
    """Use LLM to select tool with confidence scoring"""
    try:
        classification_prompt = f"""You are a tool selector. Analyze the user request and return a JSON response with your tool selection.

Available tools:
- none: Use base chat model for general questions, conversations, explanations
- wiki: Only for explicit Wikipedia requests ("from Wikipedia", "show the wiki page", "encyclopedia entry")  
- web_search: For time-sensitive info, current events, local queries, prices, availability
- code: For programming questions, debugging, code explanations
- youtube: When YouTube link provided or "summarize this video"
- read_pdf: When PDF URL or attachment detected
- desktop: For computer control ("take screenshot", "open app")

Rules:
1. Default to "none" for general conversations and questions
2. Only use wiki for EXPLICIT Wikipedia requests
3. Use web_search for current/time-sensitive information
4. Use code for programming-related queries

Return only valid JSON: {{"tool": "none", "confidence": 0.85, "reason": "General question about..."}}

User request: "{user_text}"

JSON Response:"""
        
        result = call_chat(classification_prompt, "router", code=False, autosend=False, max_tokens=150)
        
        # Try to parse JSON response
        try:
            import json
            # Extract JSON from response (may have extra text)
            json_match = re.search(r'\{.*?\}', result, re.DOTALL)
            if json_match:
                response = json.loads(json_match.group())
                tool = response.get("tool", "none")
                confidence = float(response.get("confidence", 0.5))
                reason = response.get("reason", "LLM classification")
                return tool, confidence, reason
        except Exception as parse_error:
            print(f"⚠️ JSON parse error: {parse_error}, raw: {result}")
        
        # Fallback: simple keyword matching if JSON parsing fails
        result_lower = result.lower()
        if "wiki" in result_lower:
            return "wiki", 0.7, "LLM suggested wiki (fallback)"
        elif "search" in result_lower:
            return "web_search", 0.7, "LLM suggested search (fallback)"
        elif "code" in result_lower:
            return "code", 0.7, "LLM suggested code (fallback)"
        else:
            return "none", 0.6, "LLM fallback to chat"
            
    except Exception as e:
        print(f"⚠️ LLM tool selector error: {e}")
        return "none", 0.5, f"Error fallback: {e}"

def route_message(user_text: str, msg_obj=None, confidence_threshold=0.6):
    """
    Improved routing with confidence scoring and proper fallbacks.
    Priority: explicit commands > file detection > specific tool indicators > chat default
    """
    if not user_text:
        return "none", 0.0, "Empty message"
    
    text = user_text.lower().strip()
    
    # 1. EXPLICIT SLASH COMMANDS (highest priority)
    explicit_cmd = detect_explicit_command(user_text)
    if explicit_cmd:
        log_routing_decision(user_text, explicit_cmd, 1.0, "Explicit slash command")
        return explicit_cmd, 1.0, "Explicit slash command"
    
    # 2. FILE/URL DETECTION (high priority)
    if is_attached_file(user_text, msg_obj):
        log_routing_decision(user_text, "read_pdf", 1.0, "PDF file/URL detected")
        return "read_pdf", 1.0, "PDF file/URL detected"
    
    # 3. CRITICAL SYSTEM COMMANDS (high priority)
    if any(phrase in text for phrase in ["take screenshot", "screenshot", "open app", "type this", "desktop control"]):
        log_routing_decision(user_text, "desktop", 0.9, "Desktop control keywords")
        return "desktop", 0.9, "Desktop control keywords"

    # Natural-language verify (e.g., "verify phase2", "verification phase2", "kindly verify phase 2")
    verify_keywords = ['verify', 'verification', 'check', 'confirm', 'test', 'validate']
    phase_keywords = ['phase', 'phase2', 'phase 2', 'second phase', 'phase two']
    
    has_verify = any(keyword in text for keyword in verify_keywords)
    has_phase = any(phase_keyword in text for phase_keyword in phase_keywords)
    
    if has_verify or (has_phase and any(word in text for word in ['verify', 'check', 'confirm', 'test'])):
        log_routing_decision(user_text, "verify", 0.95, "Natural-language verify detected")
        return "verify", 0.95, "Natural-language verify detected"

    # Natural-language desktop intents
    # Examples: "write hello on the desktop", "type hello", "run cmd", "open notepad", "create a file on the desktop named x"
    if (
        (" on the desktop" in text and (text.startswith("write ") or text.startswith("type ") or ("create" in text and "file" in text)))
        or text.startswith("run ")
        or text.startswith("open ")
        or text == "run cmd" or text == "cmd"
    ):
        log_routing_decision(user_text, "desktop", 0.9, "Natural-language desktop intent")
        return "desktop", 0.9, "Natural-language desktop intent"
    
    if text in ["reset", "clear chat", "reset chat", "clear history"]:
        log_routing_decision(user_text, "reset", 0.95, "Reset command detected")
        return "reset", 0.95, "Reset command detected"
        
    if ("status" in text and ("healthy" in text or "running" in text)) or text == "status":
        log_routing_decision(user_text, "status", 0.9, "Status check")
        return "status", 0.9, "Status check"
    
    # 4. SPECIFIC TOOL KEYWORDS (highest priority - must check before other patterns)
    
    # Web Search - explicit search keywords (check first!)
    search_keywords = [
        " search ", " google ", " find ", " look up ", " search for ", " find information about ",
        " current price of ", " today's news ", " recent news about ", " latest update "
    ]
    # Check if the message starts with or contains explicit search keywords
    if (text.startswith("search ") or 
        any(text.startswith(keyword.strip()) for keyword in search_keywords[:4]) or  # search, google, find, look up
        any(keyword.strip() in text for keyword in search_keywords)):
        log_routing_decision(user_text, "web_search", 0.95, "Explicit search keyword detected")
        return "web_search", 0.95, "Explicit search keyword detected"
    
    # Wikipedia - explicit wiki keywords
    wiki_keywords = [
        " wikipedia ", " from wikipedia ", " show the wiki ", " wikipedia page ", 
        " wikipedia entry ", " encyclopedia entry ", " wiki article ", " search wikipedia for "
    ]
    if (text.startswith("wiki ") or text == "wiki" or
        any(keyword.strip() in text for keyword in wiki_keywords)):
        log_routing_decision(user_text, "wiki", 0.95, "Explicit Wiki keyword detected")
        return "wiki", 0.95, "Explicit Wiki keyword detected"
    
    # Hacker News - hn keywords
    hn_keywords = ["hacker news", "hackernews", "hn story", "from hn"]
    if (text.startswith("hn ") or text == "hn" or
        any(keyword in text for keyword in hn_keywords)):
        log_routing_decision(user_text, "hn", 0.95, "Hacker News keyword detected")
        return "hn", 0.95, "Hacker News keyword detected"
    
    # arXiv - academic paper keywords
    arxiv_keywords = [" arxiv ", " academic paper ", " research paper ", " from arxiv "]
    if any(keyword.strip() in text for keyword in arxiv_keywords) or text.startswith("arxiv "):
        log_routing_decision(user_text, "arxiv", 0.95, "arXiv keyword detected")
        return "arxiv", 0.95, "arXiv keyword detected"
    
    # OpenAlex - research database keywords
    openalex_keywords = [" openalex ", " research database ", " academic database "]
    if any(keyword.strip() in text for keyword in openalex_keywords):
        log_routing_decision(user_text, "openalex", 0.95, "OpenAlex keyword detected")
        return "openalex", 0.95, "OpenAlex keyword detected"
    
    # CoinGecko - crypto keywords
    cg_keywords = ["coingecko", "crypto price", "coin price", "cryptocurrency", "bitcoin price", "eth price"]
    if (text.startswith("cg ") or text == "cg" or
        any(keyword in text for keyword in cg_keywords)):
        log_routing_decision(user_text, "cg", 0.95, "Crypto keyword detected")
        return "cg", 0.95, "Crypto keyword detected"
    
    # DEX - token trading keywords
    dex_keywords = ["token price", "defi token", "uniswap", "trading pair"]
    if (text.startswith("dex ") or text == "dex" or
        any(keyword in text for keyword in dex_keywords)):
        log_routing_decision(user_text, "dex", 0.95, "DEX keyword detected")
        return "dex", 0.95, "DEX keyword detected"
    
    # RSS - feed keywords
    rss_keywords = [" rss feed ", " news feed ", " subscribe to ", " feed url "]
    if (text.startswith("rss ") or text == "rss" or
        any(keyword.strip() in text for keyword in rss_keywords)):
        log_routing_decision(user_text, "rss", 0.95, "RSS keyword detected")
        return "rss", 0.95, "RSS keyword detected"
    
    # GitHub - repository keywords
    gh_keywords = [" github ", " github repo ", " git repository ", " source code ", " github.com "]
    if (text.startswith("gh ") or text == "gh" or
        any(keyword.strip() in text for keyword in gh_keywords)):
        log_routing_decision(user_text, "gh", 0.95, "GitHub keyword detected")
        return "gh", 0.95, "GitHub keyword detected"
    
    # Stack Overflow - programming help keywords
    so_keywords = ["stack overflow", "stackoverflow", "programming question", "coding help"]
    # Check for explicit "so" command at start, or other keywords as whole words
    if (text.startswith("so ") or text == "so" or
        any(keyword in text for keyword in so_keywords)):
        log_routing_decision(user_text, "so", 0.95, "Stack Overflow keyword detected")
        return "so", 0.95, "Stack Overflow keyword detected"
    
    # YouTube - video keywords
    yt_keywords = [" youtube ", " youtube video ", " youtube.com ", " watch video ", " video summary "]
    if (text.startswith("yt ") or text == "yt" or
        any(keyword.strip() in text for keyword in yt_keywords)):
        log_routing_decision(user_text, "yt", 0.95, "YouTube keyword detected")
        return "yt", 0.95, "YouTube keyword detected"
    
    # Code - programming keywords  
    code_keywords = [
        " code ", " debug ", " fix code ", " write code ", " programming help ",
        " function ", " python script ", " javascript function ", " code example ",
        " algorithm ", " implement ", " class ", " method ", " programming language ",
        " syntax ", " library ", " framework ", " api ", " data structure "
    ]

    # Check for explicit code patterns
    code_patterns = [
        r"write (?:a|some) code",
        r"create (?:a|an) (?:function|class|program)",
        r"implement (?:this|a|an) algorithm",
        r"debug (?:this|my) code",
        r"explain (?:this|the) code",
        r"how (?:to|do) (?:i|you) (?:write|create|implement)"
    ]

    # Check for programming keywords as separate words (with spaces)
    if any(keyword.strip() in text for keyword in code_keywords):
        log_routing_decision(user_text, "code", 0.9, "Programming keyword detected")
        return "code", 0.9, "Programming keyword detected"
        
    # Check for programming patterns
    for pattern in code_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            log_routing_decision(user_text, "code", 0.95, "Code pattern detected")
            return "code", 0.95, "Code pattern detected"
    
    # 5. DEFAULT TO CHAT (for general questions, conversations, explanations)
    log_routing_decision(user_text, "none", 0.9, "General question - using chat model")
    return "none", 0.9, "General question - using chat model"
# --- _process_read ---
def _process_read(url, user_id, chat_id):
    if not url:
        return safe_send(chat_id, "❌ No URL provided.")
    summary = uploads.parse_and_summarize_url(url, session_id=str(user_id))
    safe_send(chat_id, summary + "\n\n👉 Now you can ask questions about this content directly.")



# ---------------- Commands ----------------
@bot.message_handler(commands=["start"])
def start(m):
    bot.reply_to(
        m,
        "🤖 Dateria online.\nCommands:\n"
        "/chat, /code, /read <url>, /search <q>, /wiki <q>, /hn <q>, /arxiv <q>, "
        "/openalex <q>, /dex <symbol>, /cg <coin>, /rss <url>, /gh <q>, /so <q>, /yt <q>\n"
        "/reset, /status, /desktop <action> [args], /desktop_status, /approve <id>, /deny <id>\n"
        "/kill, /reset_kill\n\n"
        "📂 Just send me a PDF, DOCX, TXT, image, audio, or video file → I’ll parse + summarize it!"
    )


@bot.message_handler(commands=["help"])
def help_command(message):
    """Show available commands and usage information."""
    help_text = (
        "🤖 **Dateria Bot Commands**\n\n"
        "**💬 Chat & AI:**\n"
        "/chat <message> - Chat with AI assistant\n"
        "/code <request> - Generate code\n\n"
        "**🔍 Search & Research:**\n"
        "/search <query> - Web search\n"
        "/wiki <query> - Wikipedia search\n"
        "/hn <query> - Hacker News search\n"
        "/arxiv <query> - Academic papers\n"
        "/openalex <query> - Research papers\n"
        "/gh <query> - GitHub search\n"
        "/so <query> - Stack Overflow\n"
        "/yt <query> - YouTube search\n\n"
        "**💰 Finance & Data:**\n"
        "/dex <symbol> - DEX token info\n"
        "/cg <coin> - CoinGecko data\n\n"
        "**🔧 System & Verification:**\n"
        "/verify <phase2|all> - Run verification tests\n"
        "/status - System status\n"
        "/reset - Reset conversation\n"
        "/kill - Emergency stop\n"
        "/reset_kill - Reset kill switch\n\n"
        "**📱 Desktop Integration:**\n"
        "/desktop <action> - Desktop commands\n"
        "/desktop_status - Desktop status\n"
        "/approve <id> - Approve action\n"
        "/deny <id> - Deny action\n\n"
        "**📁 File Processing:**\n"
        "Send PDF, DOCX, TXT, images, audio, or video files for automatic parsing and summarization!\n\n"
        "/read <url> - Process content from URL\n"
        "/rss <url> - RSS feed processing"
    )
    safe_send(message.chat.id, help_text)


@bot.message_handler(commands=["chat"])
def chat(m):
    text = m.text.replace("/chat", "", 1).strip()
    if not text:
        return bot.reply_to(m, "Usage: /chat your message")
    looks_like_code = text.strip().startswith("```") or bool(re.search(r"\b(def|class|function|var|let|const|for|while|if)\b", text))
    call_chat(text, m.from_user.id, code=False, chat_id=m.chat.id, bot=bot, single_message=True)


@bot.message_handler(commands=["code"])
def code(m):
    text = m.text.replace("/code", "", 1).strip()
    if not text:
        return bot.reply_to(m, "Usage: /code your coding request")
    call_chat(text, m.from_user.id, code=True, chat_id=m.chat.id, bot=bot, single_message=True)


# --- Kill Switch ---
@bot.message_handler(commands=["kill"])
def kill_cmd(m):
    res = kill.trigger_kill()
    save_system_event("⚠️ Sequence killed by user via Telegram")
    safe_send(m.chat.id, f"⚠️ Kill switch activated!\n{res['note']}")


@bot.message_handler(commands=["reset_kill"])
def reset_kill_cmd(m):
    res = kill.reset_kill()
    save_system_event("✅ Kill switch reset by user via Telegram")
    safe_send(m.chat.id, f"✅ Kill switch reset.\n{res['note']}")


# --- Verification Commands ---
@bot.message_handler(commands=["verify"])
def handle_verify(message):
    """Handle verification commands."""
    if not VERIFICATION_AVAILABLE:
        safe_send(message.chat.id, "❌ Verification system not available. Run `python setup_deps.py` to install dependencies.")
        return
    
    # Support both '/verify phase2' and natural language 'verify phase2' or 'verification tool for phase 2'
    if message.text.startswith('/'):
        # Slash command - extract arguments
        command_args = _extract_query(message.text, "verify")
    else:
        # Natural language - use full text
        command_args = message.text
    
    user_id = str(message.from_user.id)
    chat_id = str(message.chat.id)
    
    # Log verification request
    save_system_event(f"🔬 Verification requested by {user_id}: {message.text}")
    
    try:
        result = handle_verify_command(command_args, user_id, chat_id)
        concise = _condense_verification_output(result.get("message", ""))
        safe_send(chat_id, concise)
    
    except Exception as e:
        error_msg = f"� Verification system error: {str(e)[:200]}...\nPlease check system logs."
        safe_send(chat_id, error_msg)
        save_system_event(f"❌ Verification error for {user_id}: {str(e)}")
        save_system_event(f"❌ Verification error for {user_id}: {str(e)}")


# --- Unified Upload Handler ---
@bot.message_handler(content_types=["document", "photo", "audio", "voice", "video"])
def handle_upload(msg):
    try:
        file = None
        filename = None
        if msg.document:
            file = bot.get_file(msg.document.file_id)
            filename = msg.document.file_name
        elif msg.photo:
            file = bot.get_file(msg.photo[-1].file_id)
            filename = f"photo_{msg.photo[-1].file_id}.jpg"
        elif msg.audio:
            file = bot.get_file(msg.audio.file_id)
            filename = msg.audio.file_name or f"audio_{msg.audio.file_id}.mp3"
        elif msg.voice:
            file = bot.get_file(msg.voice.file_id)
            filename = f"voice_{msg.voice.file_id}.ogg"
        elif msg.video:
            file = bot.get_file(msg.video.file_id)
            filename = msg.video.file_name or f"video_{msg.video.file_id}.mp4"

        if not file:
            return safe_send(msg.chat.id, "❌ Unsupported file type.")

        # download to tmp
        file_path = f"/tmp/{filename}"
        downloaded = bot.download_file(file.file_path)
        with open(file_path, "wb") as f:
            f.write(downloaded)

       # parse + summarize + save to RAG (session-aware)
        summary = uploads.parse_and_summarize(file_path, session_id=str(msg.from_user.id))
        safe_send(msg.chat.id, summary + "\n\n👉 Now you can ask me questions about this file directly.")


    except Exception as e:
        safe_send(msg.chat.id, f"⚠️ Upload error: {e}")


# ---------------- Search + Info Commands ----------------
@bot.message_handler(commands=["search"])
def search_cmd(m):
    q = _extract_query(m.text, "search")
    if not q:
        return bot.reply_to(m, "Usage: /search query")
    items = apis.search_cse(q, topk=5)
    safe_send(m.chat.id, format_items(items))


@bot.message_handler(commands=["wiki"])
def wiki_cmd(m):
    q = _extract_query(m.text, "wiki")
    if not q:
        return bot.reply_to(m, "Usage: /wiki topic")
    safe_send(m.chat.id, format_items(apis.wiki(q)))


@bot.message_handler(commands=["hn"])
def hn_cmd(m):
    q = _extract_query(m.text, "hn")
    if not q:
        return bot.reply_to(m, "Usage: /hn query")
    safe_send(m.chat.id, format_items(apis.hn(q)))


@bot.message_handler(commands=["arxiv"])
def arxiv_cmd(m):
    q = _extract_query(m.text, "arxiv")
    if not q:
        return bot.reply_to(m, "Usage: /arxiv query")
    safe_send(m.chat.id, format_items(apis.arxiv(q)))


@bot.message_handler(commands=["openalex"])
def openalex_cmd(m):
    q = _extract_query(m.text, "openalex")
    if not q:
        return bot.reply_to(m, "Usage: /openalex query")
    safe_send(m.chat.id, format_items(apis.openalex(q)))


@bot.message_handler(commands=["dex"])
def dex_cmd(m):
    q = _extract_query(m.text, "dex")
    if not q:
        return bot.reply_to(m, "Usage: /dex symbol")
    safe_send(m.chat.id, format_items(apis.dex(q)))


@bot.message_handler(commands=["cg"])
def cg_cmd(m):
    q = _extract_query(m.text, "cg")
    if not q:
        return bot.reply_to(m, "Usage: /cg coin-id (e.g. bitcoin)")
    safe_send(m.chat.id, format_items(apis.coingecko(q)))


@bot.message_handler(commands=["rss"])
def rss_cmd(m):
    url = _extract_query(m.text, "rss")
    if not url:
        return bot.reply_to(m, "Usage: /rss https://feed.xml")
    safe_send(m.chat.id, format_items(apis.rss(url)))


@bot.message_handler(commands=["gh"])
def gh_cmd(m):
    q = _extract_query(m.text, "gh")
    if not q:
        return bot.reply_to(m, "Usage: /gh query")
    safe_send(m.chat.id, format_items(apis.github(q)))


@bot.message_handler(commands=["so"])
def so_cmd(m):
    q = _extract_query(m.text, "so")
    if not q:
        return bot.reply_to(m, "Usage: /so query")
    safe_send(m.chat.id, format_items(apis.stackoverflow(q)))


@bot.message_handler(commands=["yt"])
def yt_cmd(m):
    q = _extract_query(m.text, "yt")
    if not q:
        return bot.reply_to(m, "Usage: /yt query (requires YOUTUBE_API_KEY)")
    safe_send(m.chat.id, format_items(apis.youtube(q)))


@bot.message_handler(commands=["reset"])
def reset(m):
    try:
        requests.post(f"{API}/session/reset", timeout=10, headers=HEADERS)
        safe_send(m.chat.id, "✅ Session reset.")
    except Exception as e:
        safe_send(m.chat.id, f"Error: {e}")


@bot.message_handler(commands=["status"])
def status(m):
    try:
        r = requests.get(f"{API}/health", timeout=5)
        ok = r.json().get("status") == "ok"
        safe_send(m.chat.id, "✅ API healthy." if ok else "⚠️ API not healthy.")
    except Exception as e:
        safe_send(m.chat.id, f"Error: {e}")


@bot.message_handler(commands=["time"])
def time_cmd(m):
    """Return current time with timezone info"""
    try:
        if TIMEZONE:
            current_time = datetime.datetime.now(TIMEZONE)
            time_str = current_time.strftime("%Y-%m-%d %H:%M:%S %Z")
        else:
            current_time = datetime.datetime.now()
            time_str = current_time.strftime("%Y-%m-%d %H:%M:%S UTC")
        
        safe_send(m.chat.id, f"🕒 Current time: {time_str}")
    except Exception as e:
        safe_send(m.chat.id, f"⚠️ Time error: {e}")


# ---------------- Desktop Control + Approvals ----------------
@bot.message_handler(commands=["approvals"])
def approvals_cmd(m):
    if str(m.chat.id) != str(ADMIN_CHAT_ID):
        return safe_send(m.chat.id, "⛔️ Not authorized")

    pending = [r for r in approvals._load() if r.get("status") == "pending"]
    if not pending:
        return safe_send(m.chat.id, "✅ No pending approval requests.")

    lines = []
    for r in pending:
        lines.append(
            f"📌 ID: {r['id']}\n"
            f"Action: {r['action']}\n"
            f"Args: {r['args']}\n"
            f"Requested by: {r['requested_by']}\n"
            f"Time: {r['ts']}\n"
            "----"
        )
    safe_send(m.chat.id, "\n\n".join(lines))


@bot.message_handler(commands=["desktop"])
def desktop_cmd(m):
    try:
        text = (m.text or "").strip()
        text_lower = text.lower()

        action = None
        args = {}
        multi_steps = None  # Optional macro: list of (action, args)

        if text_lower.startswith("/desktop"):
            parts = text.split(maxsplit=2)
            if len(parts) < 2:
                return safe_send(m.chat.id, "Usage: /desktop <action> [args]")
            action = parts[1]
            # Better argument parsing for desktop commands
            if len(parts) == 3:
                arg_text = parts[2]
                if action == "run" and "=" not in arg_text:
                    args["cmd"] = arg_text
                elif action == "type" and "=" not in arg_text:
                    args["text"] = arg_text
                else:
                    for kv in arg_text.split():
                        if "=" in kv:
                            k, v = kv.split("=", 1)
                            args[k] = v
        else:
            # Natural-language parsing
            # 1) Create a file on the desktop named <filename> [with content <content>]
            m_create = re.search(r"create\s+(?:a\s+)?file\s+(?:on\s+the\s+)?desktop\s+(?:named|called)\s+\"?([^\"\n]+?)\"?(?:\s+with\s+content\s+\"?(.+?)\"?)?$", text_lower, re.IGNORECASE)
            if m_create:
                filename = m_create.group(1).strip()
                content = (m_create.group(2) or "").strip()
                # Build multi-step Notepad save sequence
                multi_steps = [
                    ("run", {"cmd": "notepad"}),
                ]
                if content:
                    multi_steps.append(("type", {"text": content}))
                multi_steps.append(("hotkey", {"combo": "ctrl+s"}))
                multi_steps.append(("type", {"text": f"Desktop\\{filename}"}))
                multi_steps.append(("hotkey", {"combo": "enter"}))
            else:
                # 2) Write <content> to/into/in a file on the desktop named <filename>
                m_writefile = re.search(r"write\s+\"?(.+?)\"?\s+(?:to|into|in)\s+(?:a\s+)?file\s+(?:on\s+the\s+)?desktop\s+(?:named|called)\s+\"?([^\"\n]+?)\"?$", text, re.IGNORECASE)
                if m_writefile:
                    content = m_writefile.group(1).strip()
                    filename = m_writefile.group(2).strip()
                    multi_steps = [
                        ("run", {"cmd": "notepad"}),
                        ("type", {"text": content}),
                        ("hotkey", {"combo": "ctrl+s"}),
                        ("type", {"text": f"Desktop\\{filename}"}),
                        ("hotkey", {"combo": "enter"}),
                    ]
                else:
                    # 3) Save it as <filename> on desktop (assumes an editor is already open)
                    m_saveas = re.search(r"save\s+(?:it|this)\s+as\s+\"?([^\"\n]+?)\"?\s+(?:on\s+the\s+)?desktop", text, re.IGNORECASE)
                    if m_saveas:
                        filename = m_saveas.group(1).strip()
                        multi_steps = [
                            ("hotkey", {"combo": "ctrl+s"}),
                            ("type", {"text": f"Desktop\\{filename}"}),
                            ("hotkey", {"combo": "enter"}),
                        ]
                    else:
                        if "screenshot" in text_lower:
                            action = "screenshot"
                        elif text_lower in ("run cmd", "cmd"):
                            action = "run"; args["cmd"] = "cmd"
                        elif text_lower.startswith("run "):
                            action = "run"; args["cmd"] = text[4:].strip()
                        elif text_lower.startswith("open "):
                            # Map common app aliases
                            app = text[5:].strip()
                            if app in ("calculator", "calc"):
                                app = "calc"
                            action = "run"; args["cmd"] = app
                        elif text_lower.startswith("type "):
                            action = "type"; args["text"] = text[5:].strip()
                        elif text_lower.startswith("write "):
                            content = text[6:].strip()
                            content = re.sub(r"\s+on the desktop\s*$", "", content, flags=re.IGNORECASE)
                            action = "type"; args["text"] = content
                        else:
                            return safe_send(m.chat.id, "Usage: /desktop <action> [args] or natural: 'run notepad', 'type hello', 'take screenshot', 'create file on desktop named notes.txt with content hello'")
            if "screenshot" in text_lower:
                action = "screenshot"
            elif text_lower in ("run cmd", "cmd"):
                action = "run"; args["cmd"] = "cmd"
            elif text_lower.startswith("run "):
                action = "run"; args["cmd"] = text[4:].strip()
            elif text_lower.startswith("open "):
                action = "run"; args["cmd"] = text[5:].strip()
            elif text_lower.startswith("type "):
                action = "type"; args["text"] = text[5:].strip()
            elif text_lower.startswith("write "):
                # Remove trailing 'on the desktop' if present
                content = text[6:].strip()
                content = re.sub(r"\s+on the desktop\s*$", "", content, flags=re.IGNORECASE)
                action = "type"; args["text"] = content
            else:
                return safe_send(m.chat.id, "Usage: /desktop <action> [args] or natural: 'run notepad', 'type hello', 'take screenshot'")

        # Check if this chat ID is authorized to run desktop commands without approval
        chat_id_str = str(m.chat.id)
        if chat_id_str in AUTHORIZED_DESKTOP_CHAT_IDS or chat_id_str == str(ADMIN_CHAT_ID):
            # Authorized: run directly without policy check
            if multi_steps:
                results = []
                for (act, a) in multi_steps:
                    try:
                        r = run_desktop_command(act, a)
                        results.append({"action": act, "args": a, "result": r})
                    except Exception as step_err:
                        results.append({"action": act, "args": a, "error": str(step_err)})
                        break
                # Summarize results
                pretty = []
                for i, item in enumerate(results, 1):
                    if "error" in item:
                        pretty.append(f"{i}. {item['action']} {item['args']} -> ❌ {item['error']}")
                    else:
                        pretty.append(f"{i}. {item['action']} {item['args']} -> ✅ {item['result']}")
                safe_send(m.chat.id, "\n".join(pretty))
                return
            else:
                result = run_desktop_command(action, args)
                if action == "screenshot" and isinstance(result, dict) and "saved_to" in result:
                    path = result["saved_to"]
                    if os.path.exists(path):
                        with open(path, "rb") as f:
                            bot.send_photo(m.chat.id, f, caption=f"📸 Screenshot (via {WIN_IP}) [Authorized]")
                    else:
                        safe_send(m.chat.id, f"⚠️ Screenshot path not found: {path}")
                else:
                    safe_send(m.chat.id, f"✅ Desktop result (via {WIN_IP}) [Authorized]:\n{result}")
                return

        # For non-authorized users: use policy check
        decision = policy.check(action, args, m.from_user.id)

        if decision == "allow":
            result = run_desktop_command(action, args)
            if action == "screenshot" and isinstance(result, dict) and "saved_to" in result:
                path = result["saved_to"]
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        bot.send_photo(m.chat.id, f, caption=f"📸 Screenshot (via {WIN_IP})")
                else:
                    safe_send(m.chat.id, f"⚠️ Screenshot path not found: {path}")
            else:
                safe_send(m.chat.id, f"✅ Desktop result (via {WIN_IP}):\n{result}")

        elif decision == "ask":
            req = approvals.add_request(action, args, m.from_user.id)
            safe_send(m.chat.id, f"🔔 Approval required. Request ID: {req['id']}")
            if ADMIN_CHAT_ID:
                safe_send(
                    ADMIN_CHAT_ID,
                    f"Approval request {req['id']} from {m.from_user.id}: action={action} args={args}\nUse /approve {req['id']} or /deny {req['id']}",
                )

        else:
            safe_send(m.chat.id, "⛔️ Action denied by policy.")

    except Exception as e:
        safe_send(m.chat.id, f"❌ Desktop error (host={WIN_IP}): {e}")


@bot.message_handler(commands=["desktop_status"])
def desktop_status(m):
    try:
        res = run_desktop_command("screenshot", {})
        if isinstance(res, dict) and res.get("error"):
            safe_send(m.chat.id, f"⚠️ Desktop agent unreachable (via {WIN_IP}): {res['error']}")
        else:
            safe_send(m.chat.id, f"✅ Desktop agent reachable (via {WIN_IP})")
    except Exception as e:
        safe_send(m.chat.id, f"Error: {e}")


@bot.message_handler(commands=["approve"])
def approve_cmd(m):
    if str(m.chat.id) != str(ADMIN_CHAT_ID):
        return safe_send(m.chat.id, "⛔️ Not authorized")
    parts = m.text.split(maxsplit=1)
    if len(parts) < 2:
        return safe_send(m.chat.id, "Usage: /approve <request_id>")
    req_id = parts[1].strip()
    req = approvals.find_request(req_id)
    if not req:
        return safe_send(m.chat.id, f"No request {req_id}")
    approvals.set_status(req_id, "approved")
    safe_send(m.chat.id, f"✅ Approved {req_id}, executing now…")
    result = run_desktop_command(req["action"], req.get("args", {}))
    safe_send(m.chat.id, f"Execution result:\n{result}")


@bot.message_handler(commands=["deny"])
def deny_cmd(m):
    if str(m.chat.id) != str(ADMIN_CHAT_ID):
        return safe_send(m.chat.id, "⛔️ Not authorized")
    parts = m.text.split(maxsplit=1)
    if len(parts) < 2:
        return safe_send(m.chat.id, "Usage: /deny <request_id>")
    req_id = parts[1].strip()
    req = approvals.find_request(req_id)
    if not req:
        return safe_send(m.chat.id, f"No request {req_id}")
    approvals.set_status(req_id, "denied")
    safe_send(m.chat.id, f"❌ Denied {req_id}")

@bot.message_handler(commands=["read"])
def read_cmd(m):
    url = m.text.replace("/read", "", 1).strip()
    return _process_read(url, m.from_user.id, m.chat.id)

# ---------------- Fallback ----------------
@bot.message_handler(func=lambda m: True)
def default(m):
    try:
        # Get routing decision with confidence and reason
        mode, confidence, reason = route_message(m.text, m)
        
        # Map tool names to functions with error handling and fallbacks
        try:
            if mode == "none":
                return call_chat(m.text, m.from_user.id, code=False, chat_id=m.chat.id, bot=bot, single_message=True)
            elif mode == "code":
                return call_chat(m.text, m.from_user.id, code=True, chat_id=m.chat.id, bot=bot, single_message=True)
            elif mode == "read" or mode == "read_pdf":
                return _process_read(m.text.strip(), m.from_user.id, m.chat.id)
            elif mode == "desktop":
                return desktop_cmd(m)
            elif mode == "search" or mode == "web_search":
                return search_cmd(m)
            elif mode == "wiki":
                return wiki_cmd(m)
            elif mode == "yt" or mode == "youtube":
                return yt_cmd(m)
            elif mode == "gh":
                return gh_cmd(m)
            elif mode == "so":
                return so_cmd(m)
            elif mode == "cg":
                return cg_cmd(m)
            elif mode == "rss":
                return rss_cmd(m)
            elif mode == "reset":
                return reset(m)
            elif mode == "status":
                return status(m)
            elif mode == "time":
                return time_cmd(m)
            elif mode == "verify":
                return handle_verify(m)
            elif mode == "approvals":
                return approvals_cmd(m)
            elif mode == "desktop_status":
                return desktop_status(m)
            elif mode == "approve":
                return approve_cmd(m)
            elif mode == "deny":
                return deny_cmd(m)
            else:
                # Fallback to chat for unknown modes
                log_routing_decision(m.text, "none", 0.5, f"Unknown mode {mode}, fallback to chat")
                return call_chat(m.text, m.from_user.id, code=False, chat_id=m.chat.id, bot=bot, single_message=True)
                
        except Exception as tool_error:
            # Tool-specific error: inform user and fallback to chat
            error_msg = f"⚠️ {mode} tool failed: {str(tool_error)[:100]}... Using chat mode instead.\n\n"
            print(f"Tool error [{mode}]: {tool_error}")
            safe_send(m.chat.id, error_msg)
            return call_chat(m.text, m.from_user.id, code=False, chat_id=m.chat.id, bot=bot, single_message=True)
            
    except Exception as e:
        # Ultimate fallback: if routing fails completely, use chat
        error_msg = f"⚠️ Routing error: {e}"
        print(error_msg)
        safe_send(m.chat.id, f"🤖 Had a routing issue, using chat mode instead.\n\n")
        return call_chat(m.text, m.from_user.id, code=False, chat_id=m.chat.id, bot=bot)

if __name__ == "__main__":
    print("🤖 Telegram bot running with uploads + research tools + RAG + Desktop Control...")
    bot.infinity_polling()
