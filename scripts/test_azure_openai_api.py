from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import yaml
from openai import OpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cardiac_agent_postproc.api_client import OpenAICompatClient


def _load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _get_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key
    for env_name in (args.api_key_env, "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY"):
        value = os.getenv(env_name)
        if value:
            return value
    return ""


def _sdk_test(base_url: str, api_key: str, model: str, timeout: float) -> dict:
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Return compact JSON only: {\"answer\":\"capital of France\"}",
            }
        ],
        temperature=0.0,
    )
    text = resp.choices[0].message.content if resp.choices else ""
    return {"raw_text": text}


def _project_client_test(base_url: str, api_key: str, model: str, timeout: float) -> dict:
    client = OpenAICompatClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=timeout,
        provider="ollama",  # project's OpenAI-compatible code path
    )
    result = client.chat_json(
        system="You are a strict JSON API.",
        user='Respond with JSON only: {"answer":"capital of France"}',
        temperature=0.0,
        max_tokens=128,
    )
    return result


def _sdk_vision_test(base_url: str, api_key: str, model: str, timeout: float, image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one short sentence."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content if resp.choices else ""


def _project_client_vision_test(
    base_url: str, api_key: str, model: str, timeout: float, image_path: str
) -> dict:
    client = OpenAICompatClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=timeout,
        provider="ollama",
    )
    return client.chat_vision_json(
        system="You are a vision model. Reply with JSON only.",
        user_text='Respond as JSON: {"has_yellow_square": true/false, "brief": "..."}',
        image_path=image_path,
        temperature=0.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/azure_openai_medrag.yaml")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env", default="AZURE_OPENAI_API_KEY")
    parser.add_argument("--image", default="", help="Optional PNG/JPG path for vision test")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    _load_env_file(".env")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    llm_cfg = cfg.get("llm", {})
    base_url = args.base_url or llm_cfg.get("base_url", "")
    model = args.model or llm_cfg.get("model", "gpt-4o")
    api_key = _get_key(args)

    if not base_url:
        print("[FAIL] Missing base_url")
        return 1
    if not model:
        print("[FAIL] Missing model/deployment name")
        return 1
    if not api_key:
        print("[FAIL] Missing API key. Pass --api-key or set env AZURE_OPENAI_API_KEY/OPENAI_API_KEY.")
        return 1

    print(f"[INFO] base_url={base_url}")
    print(f"[INFO] model={model}")
    print(f"[INFO] key_loaded=yes (len={len(api_key)})")

    try:
        sdk = _sdk_test(base_url, api_key, model, args.timeout)
        print(f"[PASS] OpenAI SDK call OK: {sdk.get('raw_text', '')[:200]}")
    except Exception as e:
        print(f"[FAIL] OpenAI SDK call failed: {e}")
        return 2

    try:
        proj = _project_client_test(base_url, api_key, model, args.timeout)
        print(f"[PASS] Project OpenAICompatClient call OK: {json.dumps(proj, ensure_ascii=False)[:200]}")
    except Exception as e:
        print(f"[FAIL] Project OpenAICompatClient call failed: {e}")
        return 3

    if args.image:
        if not os.path.exists(args.image):
            print(f"[FAIL] Vision image not found: {args.image}")
            return 4
        try:
            vision_text = _sdk_vision_test(base_url, api_key, model, args.timeout, args.image)
            if not vision_text:
                print("[FAIL] OpenAI SDK vision call returned empty response")
                return 5
            print(f"[PASS] OpenAI SDK vision call OK: {vision_text[:200]}")
        except Exception as e:
            print(f"[FAIL] OpenAI SDK vision call failed: {e}")
            return 5

        try:
            vision_json = _project_client_vision_test(base_url, api_key, model, args.timeout, args.image)
            print(f"[PASS] Project OpenAICompatClient vision call OK: {json.dumps(vision_json, ensure_ascii=False)[:200]}")
        except Exception as e:
            print(f"[FAIL] Project OpenAICompatClient vision call failed: {e}")
            return 6

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
