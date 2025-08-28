import os, sys, datetime, tempfile, subprocess, collections
import psycopg, torch, requests
from urllib.parse import urlparse
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, find_dotenv



load_dotenv(find_dotenv(), override=True)

DSN = os.getenv("DATABASE_URL", "postgresql://cortex:cortex123@127.0.0.1:5440/cortex_rag")
EMB_MODEL = os.getenv("EMB_MODEL", "intfloat/multilingual-e5-large")
DEFAULT_QUERY_PATH = os.getenv(
    "DEFAULT_QUERY_PATH",
    "/home/anastasia/PycharmProjects/proposal-rag/Запрос  коммерческих предложения.doc"
)
TOPK = int(os.getenv("TOPK", "20"))


def _looks_like_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


#вспомогат. функции
def vec_literal(v):
    return "[" + ",".join(f"{x:.6f}" for x in v.tolist()) + "]"

def read_text_from_file(path):
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
                    ["soffice","--headless","--convert-to","txt:Text","--outdir",outdir,path],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                txt_path = os.path.join(outdir, os.path.splitext(os.path.basename(path))[0] + ".txt")
                return open(txt_path, "r", encoding="utf-8", errors="ignore").read()
            except Exception as e:
                raise RuntimeError("Нужен antiword или libreoffice для чтения .doc") from e
    return None


def _looks_like_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http","https") and bool(u.netloc)
    except Exception:
        return False

def _normalize_chat_url(u: str) -> str:
    u = u.rstrip("/")
    if u.endswith("/v1"):
        return u + "/chat/completions"
    if "/v1/" in u and not u.endswith("/chat/completions"):
        base = u.split("/v1/", 1)[0] + "/v1"
        return base + "/chat/completions"
    if "chat/completions" not in u:
        return u + "/v1/chat/completions"
    return u


def resolve_llm_config():
    url   = os.getenv("LLM_API_URL", "http://127.0.0.1:11434/v1/chat/completions").strip()
    key   = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "gpt-4.1").strip() or "gpt-4.1"

    if not key:
        raise RuntimeError("LLM_API_KEY не задан.")

    if not _looks_like_url(url):
        raise RuntimeError(f"LLM_API_URL некорректен: {url!r}")
    url = _normalize_chat_url(url)

    if url.lower().startswith("http://"):
        verify = None
    else:
        ca_file = os.getenv("LLM_CA_FILE", "").strip() or None
        verify = ca_file if ca_file else (os.getenv("LLM_VERIFY_SSL", "true").lower() == "true")

    return url, key, model, verify


def db_search_all(q_text_short: str, q_vec_lit: str, k: int):
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SET search_path TO rag;")
        try:
            cur.execute("""
                SELECT id, context_id, chunk_index, cos_sim, preview, document_id, section_key, section_title
                FROM rag.search_hybrid(%s, %s::vector(1024), %s, NULL::text[]);
            """, (q_text_short, q_vec_lit, max(k*6, 200)))
            rows = cur.fetchall()
            if rows:
                return rows, "HYBRID"
        except Exception as e:
            print(f"search_hybrid failed: {e}. Fallback to LEX+ANN.", file=sys.stderr)

        cur.execute("""
            WITH params AS (
              SELECT unaccent(lower(%s)) AS qt, %s::vector(1024) AS qv
            ),
            base AS (
              SELECT id, context_id, chunk_index, text_norm,
                     unaccent(lower(text_norm)) AS text_lc, embedding_1024, meta
              FROM rag.retriever_segments
            ),
            lex AS (
              SELECT id
              FROM base, params
              ORDER BY similarity(text_lc, qt) DESC
              LIMIT %s
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
        """, (q_text_short, q_vec_lit, max(k*10, 500), max(k*6, 200)))

        return cur.fetchall(), "FALLBACK"

def enrich_rows_with_doc_and_ord(rows):
    if not rows:
        return rows

    ids = [r[0] for r in rows]
    by_id = {r[0]: list(r) for r in rows}

    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SET search_path TO rag;")
        cur.execute("""
            SELECT rs.id,
                   COALESCE(rs.meta->>'document_id', lc.document_id)              AS document_id,
                   COALESCE(NULLIF(rs.meta->>'order_idx','')::int, lc.order_idx) AS order_idx,
                   COALESCE(rs.meta->>'section_key', lc.section_key)              AS section_key,
                   COALESCE(rs.meta->>'section_title', lc.section_title)          AS section_title
            FROM rag.retriever_segments rs
            LEFT JOIN rag.llm_contexts lc ON lc.id = rs.context_id
            WHERE rs.id = ANY(%s)
        """, (ids,))

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
    return [tuple(by_id[i]) for i in ids]

def diversify_by_document(rows, k, per_doc_cap=3, per_section_cap=2):
    items = []
    for r in rows:
        items.append({
            "id": r[0], "context_id": r[1], "chunk_index": r[2],
            "cos_sim": float(r[3]), "preview": r[4] or "",
            "document_id": r[5] or "UNKNOw",
            "section_key": r[6] or "", "section_title": r[7] or "",
            "order_idx": r[8] if len(r) > 8 else None
        })
    by_doc = collections.defaultdict(list)
    for it in items:
        by_doc[it["document_id"]].append(it)
    for d in by_doc:
        by_doc[d].sort(key=lambda x: x["cos_sim"], reverse=True)
    doc_order = sorted(by_doc.keys(), key=lambda d: by_doc[d][0]["cos_sim"], reverse=True)

    picked, used_per_doc = [], collections.Counter()
    used_per_doc_section = collections.defaultdict(collections.Counter)
    while len(picked) < k and doc_order:
        progressed = False
        for d in list(doc_order):
            if len(picked) >= k:
                break
            if used_per_doc[d] >= per_doc_cap:
                doc_order.remove(d); continue
            cand = None
            for it in by_doc[d]:
                if it.get("_used"):
                    continue
                sec = it["section_key"] or it["section_title"]
                if used_per_doc_section[d][sec] < per_section_cap:
                    cand = it; break
            if cand:
                cand["_used"] = True
                picked.append(cand)
                used_per_doc[d] += 1
                used_per_doc_section[d][cand["section_key"] or cand["section_title"]] += 1
                progressed = True
        if not progressed:
            break
    if len(picked) < k:
        rest = [it for it in items if not it.get("_used")]
        rest.sort(key=lambda x: x["cos_sim"], reverse=True)
        picked.extend(rest[: (k - len(picked)) ])
    return picked[:k]

#маркдаун для модели
def build_context_md(items, expand=1, per_doc_cap=8, total_char_cap=4500):
    ids = [it.get("id") for it in items if it.get("id") is not None]
    if not ids:
        return "Контекст отсутствует\n", {"docs": [], "chars": 0}

    doc_hits = {}
    import psycopg
    with psycopg.connect(DSN) as c, c.cursor() as cur:
        cur.execute("SET search_path TO rag;")
        cur.execute("""
            SELECT rs.id,
                   COALESCE(rs.meta->>'document_id', lc.document_id)              AS doc,
                   COALESCE(NULLIF(rs.meta->>'order_idx','')::int, lc.order_idx) AS ord
            FROM rag.retriever_segments rs
            LEFT JOIN rag.llm_contexts lc ON lc.id = rs.context_id
            WHERE rs.id = ANY(%s)
        """, (ids,))
        for _id, doc, ord_ in cur.fetchall():
            if doc is not None and ord_ is not None:
                doc_hits.setdefault(doc, set()).add(int(ord_))

    if not doc_hits:
        return "Контекст отсутствует\n", {"docs": [], "chars": 0}

    for doc, inds in list(doc_hits.items()):
        ext = set()
        for oi in inds:
            for d in range(-expand, expand + 1):
                if oi + d >= 0:
                    ext.add(oi + d)
        doc_hits[doc] = ext

    used_chars, parts = 0, []
    with psycopg.connect(DSN) as c, c.cursor() as cur:
        cur.execute("SET search_path TO rag;")
        for doc, inds in doc_hits.items():
            if used_chars >= total_char_cap:
                break
            want = sorted(inds)
            if not want:
                continue
            lo, hi = min(want), max(want)

            cur.execute("""
                SELECT order_idx, section_title, text_md
                FROM rag.llm_contexts
                WHERE document_id = %s AND order_idx BETWEEN %s AND %s
                ORDER BY order_idx
            """, (doc, lo, hi))
            rows = cur.fetchall()
            if not rows:
                continue

            fallback_text = {}
            empty_oids = [oi for (oi, title, txt) in rows if not (txt and txt.strip())]
            if empty_oids:
                cur.execute("""
                    SELECT (rs.meta->>'order_idx')::int AS ord,
                           rs.text_norm
                    FROM rag.retriever_segments rs
                    WHERE rs.meta->>'document_id' = %s
                      AND (rs.meta->>'order_idx')::int = ANY(%s)
                """, (doc, empty_oids))
                by_ord = {}
                for ord_, t in cur.fetchall():
                    if t:
                        if ord_ not in by_ord or len(t) > len(by_ord[ord_]):
                            by_ord[ord_] = t
                fallback_text = by_ord

            parts.append(f" {doc}")
            per_doc_used = 0
            valid = set(inds)

            for oi, title, txt in rows:
                if used_chars >= total_char_cap:
                    break
                if oi not in valid:
                    continue

                s_title = (title or "").strip().replace("\n", " ")
                md_text = (txt or "").strip()
                if not md_text:
                    md_text = (fallback_text.get(oi, "") or "").strip()
                md_text = md_text.replace("\r", "")
                if len(md_text) > 1200:
                    md_text = md_text[:1200] + " …"
                if not md_text:
                    continue

                parts.append(f"- [ctx:{doc}#{oi}] **{s_title}** — {md_text}")
                used_chars += len(md_text)
                per_doc_used += 1
                if per_doc_used >= per_doc_cap:
                    break

    if not parts:
        return "Контекст отсутствует\n", {"docs": [], "chars": 0}
    return "\n".join(parts) + "\n", {"docs": list(doc_hits.keys()), "chars": used_chars}

#функция(промт) llm
def llm_generate_kp(client_text, context_md):
    llm_url, llm_key, llm_model, verify = resolve_llm_config()
    if not llm_key:
        raise RuntimeError("LLM ключ не найден.")

    system = (
        "Ты — Principal Solutions Architect и Head of Presales компании ООО «КОРТЕКС». "
        "Твоя задача — составить <ПРОДАВАЕМОЕ> коммерческое предложение на русском языке "
        "от лица ООО «КОРТЕКС».\n"
        "Пиши так, чтобы документ убеждал и продавал: "
        "подчеркивай выгоды для клиента, а не только технические детали.\n"
        "Источники: работай ТОЛЬКО с данными из CONTEXT. "
        "Все факты подтверждай ссылкой [CTX:DOC#order]. "
        "Не выдумывай. Если информации не хватает — пиши «предварительно».\n"
        "Опыт: используй прошлые проекты компании (по названиям из section_title/title_project). "
        "Документ должен быть цельным, логичным, с понятными заголовками и таблицами.\n"
        "Контакты компании фиксированы: ООО «КОРТЕКС», 350000, г. Краснодар, ул. Северная, д. 324, 10 этаж, "
        "сайт: www.spellsystems.com."
    )

    user = f"""
CLIENT REQUEST:
{client_text.strip()}

CONTEXT:
{context_md.strip()}

# ЗАДАНИЕ
Составь ПОЛНОЦЕННОЕ коммерческое предложение ООО «КОРТЕКС» в стиле продающего КП.  
Структура документа:

1. **Титульный блок** — ООО «КОРТЕКС», контакты, «КОММЕРЧЕСКОЕ ПРЕДЛОЖЕНИЕ».
2. **Краткое резюме ценности** — в 3–4 предложениях сформулируй, какую главную задачу клиента мы решаем и почему это важно. 
   Это должно быть в самом начале (как elevator pitch).
3. **Наименование проекта** — по сути запроса.
4. **Цель проекта** — кратко, со ссылками [CTX], через призму пользы для заказчика.
5. **Основные задачи и система** — перечисление задач и функций (связать с выгодой клиента).
6. **Основные возможности** — развернутый список модулей, интерфейсов, интеграций.
7. **Технологии и методология внедрения** — стек, подходы, сроки, поэтапно.
8. **Этапы реализации** — таблица «Этап | Длительность | Комментарии».
9. **Стоимость** — таблица «Этап | Стоимость»; если нет данных — «предварительно».
10. **Варианты развертывания** — On-premise / SaaS, плюсы и минусы для клиента.
11. **Оборудование / SaaS-стоимость** — если есть данные; иначе «предварительно».
12. **Опыт реализации аналогичных проектов** — отдельный раздел. 
    Таблица: «Название проекта | Масштаб | Роль ООО «КОРТЕКС» | Результаты | [CTX]».
13. **Дополнительные услуги** — сопровождение, интеграции, обучение.
14. **Вывод и следующие шаги** — 2–3 предложения с призывом: 
    «Предлагаем обсудить детали на встрече/звонке…» и варианты контакта.

ТРЕБОВАНИЯ:
- Используй только данные из CONTEXT с [CTX].
- Проекты упоминай только по названиям (section_title/title_project).
- В каждом разделе показывай не только факты, но и смысл для клиента: как это снижает риски, экономит деньги, ускоряет внедрение.
- Текст — единый цельный документ в Markdown.
"""

    payload = {
        "model": llm_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user}
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.0,
        "max_tokens": 6000
    }
    headers = {"Authorization": f"Bearer {llm_key}", "Content-Type":"application/json"}
    resp = requests.post(llm_url, json=payload, headers=headers,
                         timeout=float(os.getenv("model_TIMEOUT","150")), verify=verify)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
def main():
    arg  = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_QUERY_PATH
    k    = int(sys.argv[2]) if len(sys.argv) >= 3 else TOPK

    _url, _key, _model, _ = resolve_llm_config()
    print(f"LLM endpoint: {_url}")
    print(f"LLM model   : {_model}")
    print(f"LLM key head: {( _key[:10] + '...' ) if _key else '<none>'}")

    q_text = read_text_from_file(arg) or arg
    q_text_short = q_text[:4000]

    # 2) эмбеддинг (e5, делим query)
    device = "cpu"
    model = SentenceTransformer(EMB_MODEL, device=device); model.max_seq_length = 512
    q_vec = model.encode(["query: " + q_text_short], normalize_embeddings=True)[0]
    q_vec_lit = vec_literal(q_vec)
    raw_rows, mode = db_search_all(q_text_short, q_vec_lit, k=max(k, TOPK))
    raw_rows = enrich_rows_with_doc_and_ord(raw_rows)
    print(f"\nRETRIEVAL: {mode}; corpus=ALL; raw_candidates={len(raw_rows)}\n")

    picked = diversify_by_document(raw_rows, k=k, per_doc_cap=3, per_section_cap=2)

    seen_docs = set()
    for it in picked:
        seen_docs.add(it["document_id"])
        ord_str = "" if it.get("order_idx") is None else f"#{it['order_idx']}"
        print(f"[{it['cos_sim']:.3f}] {it['document_id']} · CTX:{it['context_id']}{ord_str} · {it['section_title']}")
    print(f"\nCOVERAGE: {len(seen_docs)} документов в контексте ответа\n")

    context_md, cov = build_context_md(picked, expand=1, per_doc_cap=8, total_char_cap=4500)
    if cov.get('chars', 0) < 800:
        context_md, cov = build_context_md(picked, expand=2, per_doc_cap=10, total_char_cap=6000)
    print(f"CONTEXT: docs={len(cov.get('docs', []))}, chars≈{cov.get('chars', 0)}")

    try:
        kp_md = llm_generate_kp(q_text_short, context_md)
    except Exception as e:
        print("LLM недоступна:", e)
        kp_md = "Коммерческое предложение (черновик)\n\n" + context_md

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    fname = f"kp_all_{ts}.md"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(kp_md)
    print(f"\nSaved: {fname}")


if __name__ == "__main__":
    main()