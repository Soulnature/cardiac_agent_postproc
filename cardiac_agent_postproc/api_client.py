"""
Unified API client supporting multiple LLM backends:
- Google Gemini (via google-genai SDK)
- Ollama / OpenAI-compatible (via openai SDK)

Class name kept as OpenAICompatClient for backward compatibility with
the rest of the codebase — all callers import this name.
"""
from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OpenAICompatClient:
    """
    Unified LLM client supporting Gemini and Ollama/OpenAI-compatible backends.

    provider="gemini"  → uses google-genai SDK
    provider="ollama"  → uses openai SDK (OpenAI-compatible API, works with Ollama)
    """
    base_url: str       # Ollama: "http://localhost:11434/v1"; Gemini: ignored
    api_key: str        # Ollama: "ollama"; Gemini: Google AI Studio key
    model: str          # e.g. "ministral-3:14b" or "gemini-2.5-flash"
    timeout: float = 180.0
    provider: str = "gemini"  # "gemini" or "ollama"

    # lazy clients (not constructor params)
    _gemini_client: Any = field(default=None, repr=False, init=False)
    _openai_client: Any = field(default=None, repr=False, init=False)

    # ---- Client initialization ----

    def _get_gemini_client(self):
        if self._gemini_client is None:
            from google import genai
            self._gemini_client = genai.Client(api_key=self.api_key)
        return self._gemini_client

    def _get_openai_client(self):
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._openai_client

    # ---- Helpers ----

    def _reinforce_system(self, system: str) -> str:
        """Append JSON-only instruction for Ollama provider (small models need it)."""
        if self.provider == "ollama":
            suffix = "\n\nCRITICAL: You MUST respond with valid JSON only. No prose, no markdown, no explanation outside the JSON object."
            if suffix.strip() not in system:
                return system + suffix
        return system

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """Extract the first JSON object from text."""
        # Strip markdown fences
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = text.replace("```", "")

        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0), strict=False)
            except json.JSONDecodeError:
                pass

        # Narrower fallback for "score" responses
        matches = list(re.finditer(
            r'\{[^{]*"score"\s*:\s*\d+[^}]*\}', text, re.DOTALL | re.IGNORECASE
        ))
        if matches:
            try:
                return json.loads(matches[-1].group(0), strict=False)
            except json.JSONDecodeError:
                pass

        return {}

    @staticmethod
    def _encode_image_base64(image_path: str) -> str:
        """Read an image file and return its base64 encoding."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # ---- Gemini backend ----

    def _generate_gemini(
        self,
        contents: list,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 16384,
        tag: str = "LLM",
    ) -> str:
        """Core call via google-genai SDK. Returns text or empty string."""
        from google.genai import types

        client = self._get_gemini_client()

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
        )
        if system_instruction:
            config.system_instruction = system_instruction

        max_retries = 5
        for attempt in range(max_retries):
            print(f"\n[{tag}] Requesting {self.model} via google-genai...")
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )

                if response.text:
                    return response.text

                # Check for blocked / empty
                if response.candidates:
                    for cand in response.candidates:
                        if cand.content and cand.content.parts:
                            texts = [p.text for p in cand.content.parts if p.text]
                            if texts:
                                return "\n".join(texts)

                print(f"[{tag}] Warning: empty response")
                return ""

            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait = 5 * (attempt + 1)
                    print(f"[{tag}] Rate limited. Waiting {wait}s... ({attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                print(f"[{tag}] Request failed: {e}")
                return ""

        print(f"[{tag}] Max retries exhausted.")
        return ""

    # ---- Ollama / OpenAI-compatible backend ----

    def _generate_ollama(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 16384,
        tag: str = "LLM",
    ) -> str:
        """Core call via OpenAI-compatible API (Ollama). Returns text or empty string."""
        client = self._get_openai_client()
        # Newer OpenAI models (e.g. gpt-5 family) require max_completion_tokens.
        use_max_completion_tokens = self.model.startswith("gpt-5")

        max_retries = 5
        for attempt in range(max_retries):
            print(f"\n[{tag}] Requesting {self.model} via Ollama ({self.base_url})...")
            try:
                print(f"DEBUG: Calling Ollama API for {self.model}...", flush=True)
                req_kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "response_format": {"type": "json_object"},
                }
                if use_max_completion_tokens:
                    req_kwargs["max_completion_tokens"] = max_tokens
                else:
                    req_kwargs["max_tokens"] = max_tokens
                response = client.chat.completions.create(
                    **req_kwargs,
                )
                content = None
                if response.choices:
                    msg = response.choices[0].message
                    content = msg.content
                    refusal = getattr(msg, 'refusal', None)
                    if refusal:
                        print(f"[{tag}] Model refused: {refusal}")
                print(f"DEBUG: Ollama API returned. Length: {len(content) if content else 0}", flush=True)

                if content:
                    return content

                print(f"[{tag}] Warning: empty response")
                return ""

            except Exception as e:
                err_str = str(e).lower()
                if (
                    not use_max_completion_tokens
                    and "unsupported parameter" in err_str
                    and "max_tokens" in err_str
                    and "max_completion_tokens" in err_str
                ):
                    print(
                        f"[{tag}] Switching to max_completion_tokens for {self.model} "
                        "(OpenAI compatibility)."
                    )
                    use_max_completion_tokens = True
                    continue

                # Categorized error handling for Ollama
                if "connection refused" in err_str or "connect" in err_str:
                    print(
                        f"[{tag}] Connection refused — is Ollama running? "
                        f"Start with: ollama serve"
                    )
                    return ""
                if "model" in err_str and ("not found" in err_str or "does not exist" in err_str):
                    print(
                        f"[{tag}] Model not found — pull it first: "
                        f"ollama pull {self.model}"
                    )
                    return ""
                if "out of memory" in err_str or "oom" in err_str or "cuda" in err_str:
                    print(
                        f"[{tag}] GPU out of memory — try a smaller model or "
                        f"increase GPU memory. (attempt {attempt+1}/{max_retries})"
                    )
                    if attempt < max_retries - 1:
                        wait = 10 * (attempt + 1)
                        print(f"[{tag}] Waiting {wait}s for memory to free...")
                        time.sleep(wait)
                        continue
                    return ""
                if "timeout" in err_str or "timed out" in err_str:
                    print(
                        f"[{tag}] Request timed out (limit={self.timeout}s). "
                        f"Increase timeout in config or use a faster model. "
                        f"(attempt {attempt+1}/{max_retries})"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    return ""
                if "429" in str(e) or "rate" in err_str:
                    wait = 5 * (attempt + 1)
                    print(f"[{tag}] Rate limited. Waiting {wait}s... ({attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue

                print(f"[{tag}] Request failed: {e}")
                return ""

        print(f"[{tag}] Max retries exhausted.")
        return ""

    # ---- Public API (same signatures as before) ----

    def chat_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 16384,
    ) -> Dict[str, Any]:
        """Text-only chat → JSON response."""
        if self.provider == "gemini":
            contents = [user]
            text = self._generate_gemini(
                contents,
                system_instruction=system,
                temperature=temperature,
                max_tokens=max_tokens,
                tag="LLM",
            )
        else:
            messages = [
                {"role": "system", "content": self._reinforce_system(system)},
                {"role": "user", "content": user},
            ]
            text = self._generate_ollama(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tag="LLM",
            )

        if not text:
            return {}
        return self._extract_json(text)

    def chat_vision_json(
        self,
        system: str,
        user_text: str,
        image_path: str,
        temperature: float = 0.2,
        image_detail: str = "auto",
    ) -> Dict[str, Any]:
        """Multimodal chat (text + single image) → JSON response."""
        if self.provider == "gemini":
            from google.genai import types

            img_part = types.Part.from_uri(
                file_uri=image_path,
                mime_type="image/png",
            ) if image_path.startswith("gs://") else types.Part.from_bytes(
                data=open(image_path, "rb").read(),
                mime_type="image/png",
            )
            contents = [user_text, img_part]
            text = self._generate_gemini(
                contents,
                system_instruction=system,
                temperature=temperature,
                tag="LLM-Vision",
            )
        else:
            b64 = self._encode_image_base64(image_path)
            image_url_payload: Dict[str, Any] = {
                "url": f"data:image/png;base64,{b64}",
            }
            if image_detail != "auto":
                image_url_payload["detail"] = image_detail
            messages = [
                {"role": "system", "content": self._reinforce_system(system)},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": image_url_payload,
                    },
                ]},
            ]
            text = self._generate_ollama(
                messages,
                temperature=temperature,
                tag="LLM-Vision",
            )

        if not text:
            return {}
        return self._extract_json(text)

    def chat_vision_multi_json(
        self,
        system: str,
        user_text: str,
        image_paths: list,
        image_labels: list | None = None,
        temperature: float = 0.2,
        image_detail: str = "auto",
    ) -> Dict[str, Any]:
        """Multimodal chat with MULTIPLE images → JSON response."""
        if image_labels is None:
            image_labels = [None] * len(image_paths)

        if self.provider == "gemini":
            from google.genai import types

            parts: List[Any] = [user_text]
            for label, img_path in zip(image_labels, image_paths):
                if label:
                    parts.append(label)
                img_part = types.Part.from_bytes(
                    data=open(img_path, "rb").read(),
                    mime_type="image/png",
                )
                parts.append(img_part)

            text = self._generate_gemini(
                parts,
                system_instruction=system,
                temperature=temperature,
                tag="LLM-Vision-Multi",
            )
        else:
            # Build content parts for OpenAI-compatible multimodal
            content_parts: List[Dict[str, Any]] = [
                {"type": "text", "text": user_text},
            ]
            for label, img_path in zip(image_labels, image_paths):
                if label:
                    content_parts.append({"type": "text", "text": label})
                b64 = self._encode_image_base64(img_path)
                image_url_payload: Dict[str, Any] = {
                    "url": f"data:image/png;base64,{b64}",
                }
                if image_detail != "auto":
                    image_url_payload["detail"] = image_detail
                content_parts.append({
                    "type": "image_url",
                    "image_url": image_url_payload,
                })

            messages = [
                {"role": "system", "content": self._reinforce_system(system)},
                {"role": "user", "content": content_parts},
            ]
            text = self._generate_ollama(
                messages,
                temperature=temperature,
                tag="LLM-Vision-Multi",
            )

        if not text:
            return {}
        return self._extract_json(text)

    # ---- Warmup ----

    def warmup(self) -> bool:
        """
        Send a minimal request to preload the model into GPU memory.
        Only meaningful for Ollama (local models have cold-start latency).
        Returns True if the model responded successfully.
        """
        if self.provider != "ollama":
            return True

        print(f"[Warmup] Preloading {self.model} on {self.base_url}...")
        try:
            result = self.chat_json(
                system="You are a helpful assistant. Respond in JSON.",
                user='Respond with: {"status": "ok"}',
                temperature=0.0,
                max_tokens=32,
            )
            if result.get("status") == "ok":
                print(f"[Warmup] Model {self.model} ready.")
                return True
            # Model responded but not with expected content — still loaded
            print(f"[Warmup] Model responded (non-standard): {result}")
            return True
        except Exception as e:
            print(f"[Warmup] Failed to preload model: {e}")
            return False
