from __future__ import annotations
import os
from dataclasses import dataclass


def _read_bool(value: str | None, default: bool=False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class LLMSettings:
    llm_enabled: bool = False
    openai_base_url: str = "http://localhost:11434/v1"
    openai_api_key: str = "ollama"
    openai_model: str = "qwen2.5:latest"
    llm_provider: str = "gemini"  # "gemini" or "ollama"

    def __init__(self):
        env_path = os.getenv("ENV_FILE", ".env")
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, val = line.split("=", 1)
                        os.environ.setdefault(key.strip(), val.strip())
            except Exception:
                pass
        self.llm_enabled = _read_bool(os.getenv("LLM_ENABLED"), False)
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", self.openai_base_url)
        self.openai_api_key = os.getenv("OPENAI_API_KEY", self.openai_api_key)
        self.openai_model = os.getenv("OPENAI_MODEL", self.openai_model)
        self.llm_provider = os.getenv("LLM_PROVIDER", self.llm_provider)
