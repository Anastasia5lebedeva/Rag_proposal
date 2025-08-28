from __future__ import annotations

import argparse
import datetime
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer

from .config import get_settings
from .db import search_all, enrich_rows_with_doc_and_ord
from .logger import get_logger
from .utils import read_text_from_file, vec_literal

MAX_QUERY_CHARS = 4000
MODEL_MAX_SEQ_LENGTH = 512

log = get_logger(__name__)
settings = get_settings()


def embed_query(text: str) -> str:
    """Return a literal representation of the embedding for the given text."""
    model = SentenceTransformer(settings.emb_model, device="cpu")
    model.max_seq_length = MODEL_MAX_SEQ_LENGTH
    q_vec = model.encode(["query: " + text], normalize_embeddings=True)[0]
    return vec_literal(q_vec)


def retrieve_candidates(q_text_short: str, q_vec_lit: str, k: int) -> List[Dict[str, Any]]:
    rows, mode = search_all(q_text_short, q_vec_lit, k=k)
    log.info("retrieval mode: %s, raw_candidates: %s", mode, len(rows))
    rows = enrich_rows_with_doc_and_ord(rows)
    items = []
    for r in rows:
        items.append(
            {
                "id": r[0],
                "context_id": r[1],
                "chunk_index": r[2],
                "cos_sim": float(r[3]),
                "preview": r[4] or "",
                "document_id": r[5] or "UNKNOWN",
                "section_key": r[6] or "",
                "section_title": r[7] or "",
                "order_idx": r[8] if len(r) > 8 else None,
            }
        )
    return items[:k]


def save_output(text: str, prefix: str = "kp_all") -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    fname = f"{prefix}_{ts}.md"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)
    log.info("saved %s", fname)
    return fname


def run_pipeline(arg: str, k: int = 20) -> str:
    log.debug("pipeline started")
    q_text = read_text_from_file(arg) or arg
    if not q_text:
        raise ValueError("empty query text")
    q_text_short = q_text[:MAX_QUERY_CHARS]
    q_vec_lit = embed_query(q_text_short)
    candidates = retrieve_candidates(q_text_short, q_vec_lit, k=max(k, settings.topk))
    # TODO: build context and call LLM here
    context = "\n".join(c.get("preview", "") for c in candidates)
    result = "Commercial proposal (draft)\n\n" + context
    return save_output(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG pipeline")
    parser.add_argument("query", help="text or path to a file with query text")
    parser.add_argument("--k", type=int, default=settings.topk, help="top K documents")
    args = parser.parse_args()
    try:
        run_pipeline(args.query, k=args.k)
    except Exception as exc:  # noqa: BLE001
        log.error("pipeline failed: %s", exc)
        raise


if __name__ == "__main__":
    main()