## ARCHITECTURE_AUDIT

### 1. Дерево src/
- **Структура**: `src/proposal_rag/{api,config,db,models,repositories,services}` — в целом соответствует src-layout.
- **Отсутствует**: `src/proposal_rag/tests` (тесты лежат вне proposal_rag, но это допустимо).
- **Папка prompts**: правильно вынесена в `resources/prompts/`.
- **services/**: есть четкие модули: `document_processor.py`, `vector_search.py`, `llm_client.py`, `embedding_cache.py`, `prompt_registry.py`, `sync_segments.py`.
- **api/**: web-слой отделен, но отсутствует основной FastAPI router/app (есть только app.py с PromptRegistry).
- **models/**: только dto.py, в нем смешаны DTO и доменные модели, но нет явного разделения на схемы БД и API (DTO ≈ ORM, но не критично).
- **Дублирование**: есть дублирующие функции для загрузки промптов (`PromptRegistry` в api/app.py и `PromptLoader` в services/prompt_registry.py). Нужно оставить одну реализацию.
- **main/god-functions**: `cli.py` — монолитный, не вызывает пайплайн сервисов, только заглушки.

### 2. Слои и зависимости
- **api** не зависит от сервисов напрямую (нет FastAPI endpoints).
- **services** не зависят от api, что хорошо.
- **repositories**: только search_repository.py, работает с БД напрямую.
- **config**: настройки через pydantic, все ок.

### 3. Проблемы
- **Дублирование PromptRegistry/PromptLoader**.
- **Нет явного web entry-point (FastAPI app)**.
- **В cli.py нет вызова сервисных функций**.
- **В models/dto.py есть не-ASCII символы (русские слова в паттернах)**.
- **document_processor.py**: регулярки с кириллицей и латиницей в паттернах — надо вынести в prompts/ или отдельный config.

---

## PERFORMANCE

### 1. Кэш эмбеддингов
- **Ключ**: (model, content_hash), где content_hash = sha256(normalized_content).
- **Структура хранения**: Postgres таблица `rag.embedding_cache` (см. DDL).
- **API**: `get_or_compute(texts, model, compute_fn, ...)` — реализовано.
- **TTL/инвалидация**: не реализовано, но можно добавить поле expires_at или периодическую очистку по created_at.
- **Рекомендация**: добавить опциональный TTL/GC по created_at.

### 2. Индексы БД
- **Горячие поля**:
    - `embedding_cache`: (model, content_hash), created_at
    - `retriever_segments`: (context_id, chunk_index), section_key, lang
    - `llm_contexts`: (document_id, order_idx), section_key
- **DDL**:
    ```sql
    CREATE UNIQUE INDEX IF NOT EXISTS embedding_cache_model_hash_uidx ON rag.embedding_cache (model, content_hash);
    CREATE INDEX IF NOT EXISTS embedding_cache_created_idx ON rag.embedding_cache (created_at DESC);
    CREATE UNIQUE INDEX IF NOT EXISTS retriever_segments_uq ON rag.retriever_segments (context_id, chunk_index);
    CREATE INDEX IF NOT EXISTS retriever_segments_context_id_idx ON rag.retriever_segments (context_id);
    CREATE INDEX IF NOT EXISTS retriever_segments_section_key_idx ON rag.retriever_segments (section_key);
    CREATE INDEX IF NOT EXISTS retriever_segments_lang_idx ON rag.retriever_segments (lang);
    CREATE INDEX IF NOT EXISTS llm_contexts_doc_ord_idx ON rag.llm_contexts (document_id, order_idx);
    CREATE INDEX IF NOT EXISTS llm_contexts_section_key_idx ON rag.llm_contexts (section_key);
    ```
- **Обоснование**: все эти поля используются в WHERE/JOIN/ORDER BY, ускоряют поиск и обновление.

### 3. N+1, лишние I/O, повторные LLM
- **N+1**: enrich_rows_with_doc_and_ord делает один запрос по списку id — ок.
- **Лишние I/O**: нет явных повторных запросов.
- **LLM**: нет memoization для одинаковых запросов в llm_client.py — добавить кэширование на уровне chat/embed.

---

## INFRASTRUCTURE

### 1. Логирование
- **setup_logging**: реализовано, формат структурирован, stdout для контейнеров.
- **Уровни**: INFO/DEBUG/ERROR используются.
- **Ротация**: не реализована (но для stdout в контейнерах не критично).

### 2. Обработка ошибок
- **api/errors.py**: есть AppError, BadRequest, DatabaseError, ServiceError, middleware для FastAPI (но не подключен).
- **services**: try/except есть в некоторых местах, но не везде (например, llm_client.py не ловит httpx ошибки явно).
- **Рекомендация**: добавить явную обработку httpx.HTTPError в llm_client.py.

### 3. Валидация входа
- **api/schemas.py**: pydantic-модели, строгие типы, Field(...).
- **services**: не всегда валидируют вход (например, vector_search.py — только assert, лучше явные ошибки).

### 4. Конфигурация
- **BaseSettings**: реализовано, AliasChoices для EMBED_MODEL/EMB_MODEL.
- **.env**: не найден, но путь к нему прописан.
- **Магические числа**: все вынесены в settings.py.

### 5. Промпты
- **Вынесены**: все промпты лежат в resources/prompts.
- **Загрузка**: реализовано через PromptLoader/PromptRegistry, но две реализации — оставить одну.
- **Версионирование**: файлы имеют timestamp в имени — ок.

---

## MAIN_DECOMPOSITION

### 1. main/god-functions
- **cli.py**: команды index/query — только заглушки, не вызывают сервисы.
- **sync_segments.py**: main() — большой, но логика разбита на функции.
- **Нет FastAPI app**: отсутствует web entry-point.

### 2. Рефакторинг cli.py
- Разделить на:
    1. `parse_args()`
    2. `run_index(path)`
    3. `run_query(path, q, k)`
    4. Вызовы сервисов: document_processor.smart_chunk, vector_search.search_hybrid

### 3. Итоговый пайплайн (для query):
1. Валидация входа (query, top_k)
2. Получение эмбеддинга запроса
3. Поиск кандидатов (search_repository)
4. Обогащение метаданными
5. Формирование ответа

---

## PATCHES

### 1. Вынести не-ASCII паттерны из document_processor.py

```diff
--- a/src/proposal_rag/services/document_processor.py
+++ b/src/proposal_rag/services/document_processor.py
@@ -1,10 +1,16 @@
 import re
 
-import os, re, json, argparse, hashlib
-
-WS_RE = re.compile(r"[ \t\u00A0]+")
-MULTINL_RE = re.compile(r"\n{3,}")
-HDR_RE = re.compile(r"(?m)^(#{1,6}\s+.+|[А-ЯA-Z0-9][^\n]{0,80}:$|(?:Раздел|Глава|Section)\s+\d+[^\n]*$)")
-BULLET_RE = re.compile(r"(?m)^\s*(?:[-*•]|[0-9]+\.)\s+")
+import os, re, json, argparse, hashlib
+from pathlib import Path
+
+PROMPTS_DIR = Path(__file__).parent.parent.parent / "resources" / "prompts"
+with open(PROMPTS_DIR / "hdr_pattern.txt", encoding="utf-8") as f:
+    HDR_PATTERN = f.read().strip()
+with open(PROMPTS_DIR / "bullet_pattern.txt", encoding="utf-8") as f:
+    BULLET_PATTERN = f.read().strip()
+
+WS_RE = re.compile(r"[ \t\u00A0]+")
+MULTINL_RE = re.compile(r"\n{3,}")
+HDR_RE = re.compile(HDR_PATTERN)
+BULLET_RE = re.compile(BULLET_PATTERN)
```
# FILE_TO_CREATE: resources/prompts/hdr_pattern.txt
```
(?m)^(#{1,6}\s+.+|[А-ЯA-Z0-9][^\n]{0,80}:$|(?:Раздел|Глава|Section)\s+\d+[^\n]*$)
```
# FILE_TO_CREATE: resources/prompts/bullet_pattern.txt
```
(?m)^\s*(?:[-*•]|[0-9]+\.)\s+
```

---

### 2. Удалить дублирование PromptRegistry/PromptLoader

- Оставить только PromptLoader из services/prompt_registry.py.
- В api/app.py заменить PromptRegistry на PromptLoader.

```diff
--- a/src/proposal_rag/api/app.py
+++ b/src/proposal_rag/api/app.py
@@ -1,18 +1,10 @@
 from __future__ import annotations
-from functools import lru_cache
-from pathlib import Path
-from ..config.settings import get_settings
-
-
-class PromptRegistry:
-    def __init__(self, base_dir: Path) -> None:
-        self.base_dir = base_dir
-
-    @lru_cache(maxsize=128)
-    def get(self, key: str) -> str:
-        path = (self.base_dir / f"{key}.txt").resolve()
-        if not path.is_file():
-            raise FileNotFoundError(f"prompt not found: {path}")
-        return path.read_text(encoding="utf-8")
-
-
-def get_prompt_registry() -> PromptRegistry:
-    s = get_settings()
-    return PromptRegistry(base_dir=s.PROMPTS_DIR)
+from pathlib import Path
+from ..config.settings import get_settings
+from ..services.prompt_registry import PromptLoader
+
+def get_prompt_loader() -> PromptLoader:
+    s = get_settings()
+    return PromptLoader(base=s.PROMPTS_DIR)
```

---

### 3. Явная обработка ошибок в llm_client.py

```diff
--- a/src/proposal_rag/services/llm_client.py
+++ b/src/proposal_rag/services/llm_client.py
@@ -1,6 +1,7 @@
 from __future__ import annotations
 import httpx
 from tenacity import retry, wait_exponential, stop_after_attempt
+import logging
 from . import __init__  # noqa
 from ..config.settings import get_settings
 
 s = get_settings()
+log = logging.getLogger(__name__)
 
 class LLMClient:
     def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None):
         self.base_url = (base_url or s.LLM_API_URL).rstrip("/")
         self.api_key = api_key or s.LLM_API_KEY
         self.model = model or s.LLM_MODEL
         self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
         self.timeout = s.LLM_TIMEOUT_S
 
     @retry(wait=wait_exponential(1, 10), stop=stop_after_attempt(3))
     async def chat(self, messages: list[dict]) -> str:
         url = f"{self.base_url}/chat/completions"
         payload = {"model": self.model, "messages": messages, "temperature": 0.1}
-        async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as cli:
-            r = await cli.post(url, json=payload)
-            r.raise_for_status()
-            data = r.json()
-            return data["choices"][0]["message"]["content"].strip()
+        try:
+            async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as cli:
+                r = await cli.post(url, json=payload)
+                r.raise_for_status()
+                data = r.json()
+                return data["choices"][0]["message"]["content"].strip()
+        except httpx.HTTPError as e:
+            log.error("LLMClient.chat HTTP error: %s", e)
+            raise
 
     async def embed(self, texts: list[str]) -> list[list[float]]:
         url = f"{self.base_url}/embeddings"
         payload = {"model": s.EMBED_MODEL, "input": texts}
-        async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as cli:
-            r = await cli.post(url, json=payload)
-            r.raise_for_status()
-            data = r.json()
-            # поддержка как батча, так и единичного входа
-            if isinstance(texts, str):
-                return data["data"][0]["embedding"]
-            return [d["embedding"] for d in data["data"]]
+        try:
+            async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as cli:
+                r = await cli.post(url, json=payload)
+                r.raise_for_status()
+                data = r.json()
+                # поддержка как батча, так и единичного входа
+                if isinstance(texts, str):
+                    return data["data"][0]["embedding"]
+                return [d["embedding"] for d in data["data"]]
+        except httpx.HTTPError as e:
+            log.error("LLMClient.embed HTTP error: %s", e)
+            raise
```

---

### 4. cli.py: подключить сервисы

```diff
--- a/src/proposal_rag/cli.py
+++ b/src/proposal_rag/cli.py
@@ -1,17 +1,27 @@
 from __future__ import annotations
 import asyncio
 from pathlib import Path
 import typer
+from proposal_rag.services.document_processor import smart_chunk
+from proposal_rag.services.vector_search import search_hybrid
 
 app = typer.Typer(add_completion=False)
 
 @app.command()
 def index(path: str):
     p = Path(path)
-    # вызови из твоего document_processor разбор и чанкинг
-    typer.echo(f"Indexed: {p}")
+    text = p.read_text(encoding="utf-8")
+    chunks = smart_chunk(text)
+    typer.echo(f"Indexed: {p}, {len(chunks)} chunks")
 
 @app.command()
 def query(path: str, q: str, k: int = 10):
     p = Path(path)
-    # вызови твой ретрив из vector_search
-    res = []
-    for r in res:
-        typer.echo(r)
+    # HYPOTHESIS: path не используется, только q и k
+    try:
+        hits = search_hybrid(q, k)
+        for r in hits:
+            typer.echo(r)
+    except Exception as e:
+        typer.echo(f"Error: {e}")
 
 if __name__ == "__main__":
     app()
```

---

## CHECKLIST

### 1–2 дня
- [ ] Вынести все не-ASCII паттерны из document_processor.py в resources/prompts/.
- [ ] Удалить дублирующий PromptRegistry, оставить PromptLoader.
- [ ] В cli.py подключить реальные сервисы (smart_chunk, search_hybrid).
- [ ] Добавить обработку httpx ошибок в llm_client.py.
- [ ] Проверить, что все промпты и паттерны загружаются из файлов.
- [ ] Проверить, что все индексы в БД созданы (см. DDL выше).

### 1 неделя
- [ ] Добавить FastAPI entry-point с подключением ошибок и логирования.
- [ ] Покрыть тестами cli и сервисы (unit/integration).
- [ ] Добавить TTL/GC для embedding_cache (по created_at).
- [ ] Вынести все magic numbers в settings.py.
- [ ] Провести нагрузочное тестирование поиска и кэша.

### Метрики и быстрая проверка успеха
- Все паттерны и промпты — только в resources/prompts.
- Нет кириллицы/не-ASCII в .py.
- Кэш эмбеддингов работает (проверить повторный вызов).
- Индексы в БД присутствуют (EXPLAIN ANALYZE на hot queries).
- Логирование структурировано, ошибки ловятся.
- cli index/query работают на реальных данных.

---

**Если нужны дополнительные патчи — присылай новые исходники.**