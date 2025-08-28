from __future__ import annotations

import os
import subprocess
import tempfile
from urllib.parse import urlparse
from typing import Optional


def looks_like_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in {"http", "https"} and bool(u.netloc)
    except Exception:
        return False


def normalize_chat_url(u: str) -> str:
    u = u.rstrip("/")
    if u.endswith("/v1"):
        return u + "/chat/completions"
    if "/v1/" in u and not u.endswith("/chat/completions"):
        base = u.split("/v1/", 1)[0] + "/v1"
        return base + "/chat/completions"
    if "chat/completions" not in u:
        return u + "/v1/chat/completions"
    return u


def vec_literal(v):
    return "[" + ",".join(f"{x:.6f}" for x in v.tolist()) + "]"


def read_text_from_file(path: str) -> Optional[str]:
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return open(path, "r", encoding="utf-8", errors="ignore").read()
    if ext == ".docx":
        from docx import Document
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    if ext == ".doc":
        try:
            out = subprocess.run(["antiword", path], capture_output=True, check=True)
            return out.stdout.decode("utf-8", errors="ignore")
        except Exception:
            outdir = tempfile.gettempdir()
            try:
                subprocess.run(
                    [
                        "soffice",
                        "--headless",
                        "--convert-to",
                        "txt:Text",
                        "--outdir",
                        outdir,
                        path,
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                txt_path = os.path.join(
                    outdir, os.path.splitext(os.path.basename(path))[0] + ".txt"
                )
                return open(txt_path, "r", encoding="utf-8", errors="ignore").read()
            except Exception as e:
                raise RuntimeError("antiword or libreoffice required for .doc") from e
    return None