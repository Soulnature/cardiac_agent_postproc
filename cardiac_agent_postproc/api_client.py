from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import httpx

@dataclass
class OpenAICompatClient:
    base_url: str
    api_key: str
    model: str
    timeout: float = 60.0

    def chat_json(self, system: str, user: str, temperature: float=0.2, max_tokens: int=800) -> Dict[str, Any]:
        # OpenAI-compatible Chat Completions endpoint
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"}
        }
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        # OpenAI format: choices[0].message.content
        content = data["choices"][0]["message"]["content"]
        import json
        return json.loads(content)
