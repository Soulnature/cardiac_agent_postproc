#!/usr/bin/env python
"""
Test: Use VLM to READ the quantitative stats from the diff panel header,
then make the judgment based on the numbers rather than visual interpretation.

The diff panel already shows "+283 -0 ~321" — if the model can read these
numbers, we can make decisions based on added vs removed ratio.
"""
import base64, json, requests

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
MODEL = "ministral-3:14b"

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def call_vlm(system, content_parts, temp=0.1):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content_parts},
        ],
        "temperature": temp,
        "max_tokens": 500,
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ── Quantitative prompt ──
QUANT_PROMPT = """This 3-panel image shows a cardiac MRI segmentation repair.
The RIGHT panel header shows statistics: +N (pixels added), -N (removed), ~N (class changed).

Step 1: Read the exact numbers from the DIFF panel header.
Step 2: Look at the BEFORE (left) and AFTER (middle) panels.
Step 3: Judge:
  - If mostly green additions filling gaps with few removals → IMPROVED
  - If mostly red removals destroying structure → DEGRADED  
  - If the yellow class-change pixels outnumber useful green additions → DEGRADED
  - If changes are < 50 total pixels → NEUTRAL

Respond in JSON:
{"pixels_added": <int>, "pixels_removed": <int>, "pixels_class_changed": <int>,
 "verdict": "improved"|"degraded"|"neutral",
 "reasoning": "one sentence"}"""

import os, glob

# Test with multiple images across categories
test_images = []
for cat in ["improved", "degraded"]:
    imgs = sorted(glob.glob(f"repair_kb/{cat}/*.png"))
    test_images.extend([(cat, img) for img in imgs[:3]])

for true_cat, img_path in test_images:
    fname = os.path.basename(img_path)
    b64 = encode_image(img_path)
    content = [
        {"type": "text", "text": QUANT_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]
    result = call_vlm("You are a precise image analysis assistant.", content)
    
    # Try to extract verdict
    try:
        # Find JSON in result
        import re
        json_match = re.search(r'\{[^}]+\}', result.replace('\n', ' '))
        if json_match:
            parsed = json.loads(json_match.group())
            verdict = parsed.get("verdict", "?")
            added = parsed.get("pixels_added", "?")
            removed = parsed.get("pixels_removed", "?")
            changed = parsed.get("pixels_class_changed", "?")
        else:
            verdict = "parse_error"
            added = removed = changed = "?"
    except:
        verdict = "parse_error"
        added = removed = changed = "?"
    
    correct = "✅" if verdict == true_cat else "❌"
    print(f"  {correct} TRUE={true_cat:8s} VLM={verdict:8s}  +{added} -{removed} ~{changed}  ({fname[:50]})")
