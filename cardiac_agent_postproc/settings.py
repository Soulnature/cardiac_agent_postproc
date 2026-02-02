from __future__ import annotations
from pydantic_settings import BaseSettings
from pydantic import Field

class LLMSettings(BaseSettings):
    llm_enabled: bool = Field(default=False, alias="LLM_ENABLED")
    openai_base_url: str = Field(default="http://localhost:11434/v1", alias="OPENAI_BASE_URL")
    openai_api_key: str = Field(default="ollama", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="qwen2.5:latest", alias="OPENAI_MODEL")

    class Config:
        env_file = ".env"
        extra = "ignore"
