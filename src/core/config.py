from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    database_url: str
    emb_model: str = "intfloat/multilingual-e5-large"
    default_query_path: str = ""
    topk: int = 20
    llm_api_url: str = "http://127.0.0.1:11434/v1/chat/completions"
    llm_api_key: str = ""
    llm_model: str = "gpt-4.1"
    llm_ca_file: Optional[str] = None
    llm_verify_ssl: bool = True


def _bool_env(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).lower() == "true"


def get_settings() -> Settings:
    """Load settings from environment with basic validation."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")

    llm_api_key = os.getenv("LLM_API_KEY")
    if not llm_api_key:
        raise RuntimeError("LLM_API_KEY is required")

    return Settings(
        database_url=database_url,
        emb_model=os.getenv("EMB_MODEL", Settings.emb_model),
        default_query_path=os.getenv("DEFAULT_QUERY_PATH", Settings.default_query_path),
        topk=int(os.getenv("TOPK", Settings.topk)),
        llm_api_url=os.getenv("LLM_API_URL", Settings.llm_api_url),
        llm_api_key=llm_api_key,
        llm_model=os.getenv("LLM_MODEL", Settings.llm_model),
        llm_ca_file=os.getenv("LLM_CA_FILE"),
        llm_verify_ssl=_bool_env("LLM_VERIFY_SSL", "true"),
    )