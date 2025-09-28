import os, json, csv, re
from typing import List, Dict, Any
from pypdf import PdfReader
import docx2txt
from bs4 import BeautifulSoup
import markdown as md

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path: str) -> List[Dict[str, Any]]:
    # Guard: ensure file is really a PDF
    with open(path, "rb") as fh:
        head = fh.read(5)
    if head != b"%PDF-":
        raise ValueError(f"Invalid PDF header for {path!r}: {head!r}")

    reader = PdfReader(path)
    pages = []
    for i, p in enumerate(reader.pages):
        try:
            txt = p.extract_text() or ""
        except Exception:
            txt = ""
        pages.append({"page": i+1, "text": normalize_text(txt)})
    return pages


def read_docx(path: str) -> str:
    txt = docx2txt.process(path) or ""
    return normalize_text(txt)

def read_md(path: str) -> str:
    text = read_txt(path)
    html = md.markdown(text)
    soup = BeautifulSoup(html, "html.parser")
    return normalize_text(soup.get_text("\n"))

def read_json(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return normalize_text(read_txt(path))
    lines = []
    def flatten(prefix, obj):
        if isinstance(obj, dict):
            for k,v in obj.items():
                flatten(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(obj, list):
            for i,v in enumerate(obj):
                flatten(f"{prefix}[{i}]", v)
        else:
            lines.append(f"{prefix}: {obj}")
    flatten("", data)
    return normalize_text("\n".join(lines))

def read_csv(path: str, max_rows: int = 5000) -> str:
    lines = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for i,row in enumerate(reader):
            if i == 0:
                header = ", ".join(row)
                lines.append(f"HEADERS: {header}")
            else:
                lines.append(", ".join(row))
            if i >= max_rows:
                break
    return normalize_text("\n".join(lines))

def normalize_text(t: str) -> str:
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()

def extract_any(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".txt"]:
        return {"type":"txt","text":read_txt(path)}
    if ext in [".pdf"]:
        return {"type":"pdf","pages":read_pdf(path)}
    if ext in [".docx"]:
        return {"type":"docx","text":read_docx(path)}
    if ext in [".md",".markdown"]:
        return {"type":"md","text":read_md(path)}
    if ext in [".json",".jsonl"]:
        return {"type":"json","text":read_json(path)}
    if ext in [".csv"]:
        return {"type":"csv","text":read_csv(path)}
    return {"type":"txt","text":read_txt(path)}
