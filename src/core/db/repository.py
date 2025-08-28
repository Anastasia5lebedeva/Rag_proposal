from __future__ import annotations
from typing import Iterable, Tuple, List, Any
import psycopg
from .config import get_settings
from .logger import get_logger

log = get_logger(__name__)
settings = get_settings()


def search_all(q_text_short: str, q_vec_lit: str, k: int) -> Tuple[List[Tuple[Any, ...]], str]:
    """Search the database using hybrid search with fallback."""
    try:
        with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
            cur.execute("SET search_path TO rag;")
            try:
                cur.execute(
                    """
                    SELECT id, context_id, chunk_index, cos_sim, preview, document_id, section_key, section_title
                    FROM rag.search_hybrid(%s, %s::vector(1024), %s, NULL::text[]);
                    """,
                    (q_text_short, q_vec_lit, max(k * 6, 200)),
                )
                rows = cur.fetchall()
                if rows:
                    return rows, "HYBRID"
            except Exception as e:  # noqa: BLE001
                log.warning("search_hybrid failed: %s. Fallback to LEX+ANN", e)

            cur.execute(
                """
                WITH params AS (
                  SELECT unaccent(lower(%s)) AS qt, %s::vector(1024) AS qv
                ),
                base AS (
                  SELECT id, context_id, chunk_index,
                         unaccent(lower(text_norm)) AS text_lc, embedding_1024, meta
                  FROM rag.retriever_segments
                ),
                lex AS (
                  SELECT id FROM base, params ORDER BY similarity(text_lc, qt) DESC LIMIT %s
                )
                SELECT b.id, b.context_id, b.chunk_index,
                       1 - (b.embedding_1024 <=> p.qv) AS cos_sim,
                       left(b.text_norm, 300) AS preview,
                       b.meta->>'document_id'  AS document_id,
                       b.meta->>'section_key'  AS section_key,
                       b.meta->>'section_title' AS section_title
                FROM lex
                JOIN base b USING(id)
                JOIN params p ON true
                ORDER BY b.embedding_1024 <=> p.qv
                LIMIT %s;
                """,
                (q_text_short, q_vec_lit, max(k * 10, 500), max(k * 6, 200)),
            )
            return cur.fetchall(), "FALLBACK"
    except psycopg.Error as e:
        log.error("database search failed: %s", e)
        raise


def enrich_rows_with_doc_and_ord(rows: Iterable[Tuple[Any, ...]]) -> List[Tuple[Any, ...]]:
    if not rows:
        return list(rows)

    ids = [r[0] for r in rows]
    by_id = {r[0]: list(r) for r in rows}

    try:
        with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
            cur.execute("SET search_path TO rag;")
            cur.execute(
                """
                SELECT rs.id,
                       COALESCE(rs.meta->>'document_id', lc.document_id)              AS document_id,
                       COALESCE(NULLIF(rs.meta->>'order_idx','')::int, lc.order_idx) AS order_idx,
                       COALESCE(rs.meta->>'section_key', lc.section_key)              AS section_key,
                       COALESCE(rs.meta->>'section_title', lc.section_title)          AS section_title
                FROM rag.retriever_segments rs
                LEFT JOIN rag.llm_contexts lc ON lc.id = rs.context_id
                WHERE rs.id = ANY(%s)
                """,
                (ids,),
            )
            for rid, doc, ord_, skey, stitle in cur.fetchall():
                row = by_id.get(rid)
                if not row:
                    continue
                row[5] = doc
                row[6] = skey
                row[7] = stitle
                if len(row) == 8:
                    row.append(ord_)
                else:
                    row[8] = ord_
    except psycopg.Error as e:
        log.error("failed to enrich rows: %s", e)
        raise

    return [tuple(by_id[i]) for i in ids]