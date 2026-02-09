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
    timeout: float = 180.0

    def chat_json(self, system: str, user: str, temperature: float=0.2, max_tokens: int=16384) -> Dict[str, Any]:
        """
        Sends a chat request and expects a JSON response via requests.
        """
        import requests
        import re
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False 
        }
        
        url = f"{self.base_url}/chat/completions"
        if url.endswith("//chat/completions"):
             url = url.replace("//chat", "/chat")

        print(f"\n[LLM] Requesting {self.model} via requests...")
        
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = json.loads(resp.text, strict=False)

            msg = data["choices"][0]["message"]
            content = msg.get("content")
            reasoning = msg.get("reasoning_content") or msg.get("reasoning")

            if reasoning:
                print(f"[LLM] Reasoning: {reasoning[:200]}...")

            if content is None:
                if reasoning:
                    print("[LLM] Warning: 'content' is None but 'reasoning' found. Using reasoning as content fallback.")
                    content = reasoning
                else:
                    return {}

            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                 try:
                     return json.loads(match.group(0), strict=False)
                 except:
                     pass

            # Try parsing directly
            try:
                return json.loads(content, strict=False)
            except:
                pass

            return {}
                
        except Exception as e:
            print(f"[LLM] Request failed: {e}")
            return {}

    def chat_vision_json(self, system: str, user_text: str, image_path: str, temperature: float=0.2) -> Dict[str, Any]:
        """
        Sends a multimodal chat request (Text + Image) via requests.
        Expects JSON response.
        """
        import requests
        import base64
        import re
        
        def encode_image(path):
            with open(path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        base64_image = encode_image(image_path)
        
        # ... (messages setup remains)
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 16384,
            "temperature": temperature,
            "stream": False
        }

        url = f"{self.base_url}/chat/completions"
        if url.endswith("//chat/completions"):
             url = url.replace("//chat", "/chat")

        import time as _time

        max_retries = 5
        for attempt in range(max_retries):
            print(f"\n[LLM-Vision] Requesting {self.model} via requests...")
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code == 429:
                    wait = 5 * (attempt + 1)
                    print(f"[LLM-Vision] Rate limited (429). Waiting {wait}s... ({attempt+1}/{max_retries})")
                    _time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = json.loads(resp.text, strict=False)

                msg = data["choices"][0]["message"]
                content = msg.get("content")
                reasoning = msg.get("reasoning_content") or msg.get("reasoning")

                if reasoning:
                    print(f"[LLM-Vision] Reasoning: {reasoning[:200]}...")

                if content is None:
                    if reasoning:
                        print("[LLM-Vision] Warning: 'content' is None. Fallback to reasoning.")
                        content = reasoning
                    else:
                        print(f"[LLM-Vision] Warning: 'content' is None.")
                        return {}

                # Try to find JSON block — generic approach
                # First try: find outermost {...} block
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(0), strict=False)
                    except json.JSONDecodeError:
                        pass

                # Fallback: narrower match for "score" responses
                matches = list(re.finditer(r'\{[^{]*"score"\s*:\s*\d+[^}]*\}', content, re.DOTALL | re.IGNORECASE))
                if matches:
                    try:
                        return json.loads(matches[-1].group(0), strict=False)
                    except json.JSONDecodeError:
                        pass

                return {}

            except Exception as e:
                if "429" in str(e):
                    wait = 5 * (attempt + 1)
                    print(f"[LLM-Vision] Rate limited. Waiting {wait}s... ({attempt+1}/{max_retries})")
                    _time.sleep(wait)
                    continue
                print(f"[LLM-Vision] Request failed: {e}")
                if 'resp' in locals():
                    print(f"[LLM-Vision] Response: {resp.text[:200]}")
                return {}

        print("[LLM-Vision] Max retries exhausted.")
        return {}

