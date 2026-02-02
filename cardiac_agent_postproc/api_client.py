from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import json
import sys
from openai import OpenAI

@dataclass
class OpenAICompatClient:
    base_url: str
    api_key: str
    model: str
    timeout: float = 60.0

    def chat_json(self, system: str, user: str, temperature: float=0.2, max_tokens: int=800) -> Dict[str, Any]:
        """
        Sends a chat request and expects a JSON response.
        Supports 'thinking' capability if available (prints reasoning to stdout).
        """
        client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout
        )

        # Qwen/NVIDIA specific: enable thinking via extra_body
        extra_body = {"chat_template_kwargs": {"thinking": True}}
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        print(f"\n[LLM] Requesting {self.model} (stream=True)...")
        
        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
                stream=True,
                response_format={"type": "json_object"}
            )
        except Exception as e:
            # Fallback: some models/backends might not support extra_body or json_object with thinking
            print(f"[LLM] Warning: Standard call failed ({e}). Retrying without extra_body/json constraint...")
            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )

        full_content = []
        print("[LLM] Reasoning path: ", end="", flush=True)
        
        has_reasoning = False
        
        for chunk in completion:
            # Handle reasoning content (DeepSeek/Qwen style)
            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'reasoning_content'):
                reasoning = chunk.choices[0].delta.reasoning_content
                if reasoning:
                    if not has_reasoning:
                        has_reasoning = True
                    print(reasoning, end="", flush=True)
            
            # Handle standard content
            if chunk.choices and chunk.choices[0].delta.content is not None:
                content_chunk = chunk.choices[0].delta.content
                full_content.append(content_chunk)

        print("\n[LLM] Reasoning finished.")
        
        final_str = "".join(full_content)
        
        # Clean up potential markdown code blocks if the model wrapped the JSON
        if "```json" in final_str:
            final_str = final_str.split("```json")[1].split("```")[0].strip()
        elif "```" in final_str:
            final_str = final_str.split("```")[1].split("```")[0].strip()

        try:
            return json.loads(final_str)
        except json.JSONDecodeError:
            print(f"[LLM] Failed to parse JSON. Raw output:\n{final_str}")
            # simple fallback mechanism
            return {}
