# tools/media.py
import io, pdfplumber, pytesseract, trafilatura, subprocess, tempfile, os
from PIL import Image
import requests
import whisper

# ---------------- Web Pages ----------------
def read_url(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return "Failed to fetch URL."
    text = trafilatura.extract(downloaded, include_comments=False, include_formatting=False) or ""
    return text.strip()[:12000] or "No extractable text."

# ---------------- PDFs ----------------
def read_pdf_bytes(data: bytes) -> str:
    try:
        out = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                out.append(page.extract_text() or "")
        text = "\n".join(out).strip()
        if text:
            return text[:12000]
    except Exception:
        pass
    # OCR fallback
    try:
        images = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                img = page.to_image(resolution=200).original
                images.append(img)
        ocr_text = []
        for im in images:
            ocr_text.append(pytesseract.image_to_string(Image.fromarray(im)))
        return ("\n".join(ocr_text)).strip()[:12000] or "No text via OCR."
    except Exception:
        return "PDF parsing failed."

# ---------------- Images ----------------
def read_image_bytes(data: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(data))
        txt = pytesseract.image_to_string(img)
        return txt.strip()[:12000] or "No text detected in image."
    except Exception as e:
        return f"Image OCR failed: {e}"

# ---------------- Audio / Video ----------------
def transcribe_media_bytes(data: bytes, is_video: bool = False) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4" if is_video else ".wav") as tmp:
            tmp.write(data)
            tmp.flush()
            tmp_path = tmp.name

        # If video, extract audio first
        if is_video:
            audio_path = tmp_path.replace(".mp4", ".wav")
            cmd = ["ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", audio_path]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(tmp_path)
        else:
            audio_path = tmp_path

        # Transcribe with whisper (tiny for demo, can set base/small/medium/large)
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        os.remove(audio_path)
        return result.get("text", "").strip()[:12000] or "No transcript."
    except Exception as e:
        return f"Transcription failed: {e}"
