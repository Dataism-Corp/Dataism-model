import os
import subprocess
import pytesseract
import whisper
import docx
import fitz  # PyMuPDF
from PIL import Image
import requests
from bs4 import BeautifulSoup
import tempfile

from rag_store import add_documents, search_documents

# load Whisper once
whisper_model = whisper.load_model("base")

# --------- Helpers ----------
def chunk_text(text, size=1500):
    return [text[i:i+size] for i in range(0, len(text), size)]

def save_to_rag(text, ref, ext, session_id="default"):
    """
    Save extracted text into RAG with session binding.
    ref = path or url
    """
    chunks = chunk_text(text)
    docs = [{
        "title": f"{os.path.basename(ref) if os.path.exists(ref) else ref} chunk {i+1}",
        "snippet": chunk,
        "link": ref,
        "source": f"upload-{ext}"
    } for i, chunk in enumerate(chunks)]
    add_documents(docs, session_id=session_id)

# --------- Parsers ----------
def parse_pdf(path):
    text = ""
    doc = fitz.open(path)
    for page in doc:
        text += page.get_text()
    return text

def parse_docx(path):
    text = ""
    d = docx.Document(path)
    for p in d.paragraphs:
        text += p.text + "\n"
    return text

def parse_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def parse_image(path):
    img = Image.open(path)
    return pytesseract.image_to_string(img)

def parse_audio(path):
    result = whisper_model.transcribe(path)
    return result["text"]

def parse_video(path):
    audio_path = "/tmp/extracted_audio.wav"
    subprocess.run([
        "ffmpeg", "-i", path, "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", audio_path, "-y"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return parse_audio(audio_path)

def parse_url(url: str) -> str:
    if url.lower().endswith(".pdf"):
        r = requests.get(url, timeout=30)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(r.content)
            tmp.flush()
            text = parse_pdf(tmp.name)
        return text

    # Otherwise assume webpage
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(r.text, "html.parser")

    for script in soup(["script", "style"]):
        script.decompose()
    text = " ".join(soup.stripped_strings)
    return text

# --------- Unified ----------
def parse_and_summarize(path: str, session_id="default") -> str:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        text = parse_pdf(path)
    elif ext == ".docx":
        text = parse_docx(path)
    elif ext in [".txt", ".md"]:
        text = parse_txt(path)
    elif ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        text = parse_image(path)
    elif ext in [".mp3", ".wav", ".m4a"]:
        text = parse_audio(path)
    elif ext in [".mp4", ".mov", ".avi", ".mkv"]:
        text = parse_video(path)
    else:
        return f"⚠️ Unsupported file type: {ext}"

    if not text.strip():
        return "⚠️ No readable content extracted."

    save_to_rag(text, path, ext, session_id=session_id)

    preview = text[:800].replace("\n", " ")
    return f"✅ {os.path.basename(path)} processed and stored in memory (session: {session_id}).\n\nPreview:\n{preview}..."

def parse_and_summarize_url(url: str, session_id="default") -> str:
    try:
        text = parse_url(url)
    except Exception as e:
        return f"⚠️ Failed to fetch {url}: {e}"

    if not text.strip():
        return f"⚠️ No content extracted from {url}"

    save_to_rag(text, url, "url", session_id=session_id)

    preview = text[:800].replace("\n", " ")
    return f"✅ Content from {url} processed and stored in memory (session: {session_id}).\n\nPreview:\n{preview}..."

def retrieve_context(query: str, session_id="default", topk: int = 5) -> str:
    """Search RAG store for relevant content to the query."""
    try:
        results = search_documents(query, session_id=session_id, topk=topk)
        if not results:
            return ""
        context = "\n\n".join([f"[{r.get('title','')}] {r.get('snippet','')}" for r in results])
        return context
    except Exception as e:
        return f"⚠️ RAG search error: {e}"
