import base64, json, requests

def encode_image(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

overlay_path = 'results/Input_UKB/single_panel_refs/2370447_2_original_lax_4c_035_overlay.png'
img_b64 = encode_image(overlay_path)

prompt = (
    "You are a cardiac MRI segmentation quality expert.\n\n"
    "This is a 4-chamber (4CH) cardiac MRI overlay image. Color coding:\n"
    "- Red = RV blood pool\n"
    "- Green = LV myocardium (Myo)\n"
    "- Blue = LV blood pool\n\n"
    "Please assess the segmentation quality and identify any structural problems.\n"
    "Rate quality 1-10 and classify as: good / borderline / bad\n\n"
    'Respond in JSON: {"quality": "good|borderline|bad", "score": 1-10, "confidence": 0-1, "issues": [...], "reasoning": "..."}'
)

payload = {
    'model': 'ministral-3:14b',
    'messages': [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': prompt},
            {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{img_b64}'}}
        ]
    }],
    'temperature': 0.1
}

resp = requests.post('http://localhost:11434/v1/chat/completions', json=payload, timeout=300)
result = resp.json()
content = result['choices'][0]['message']['content']
print('=== VLM Raw Response ===')
print(content)

# 也测试一个好的 case 作对比
print('\n\n=== Now testing a GOOD case for comparison ===')
# 从之前实验里找一个 dice~0.93+ 的 case
good_path = 'results/Input_UKB/single_panel_refs/2370447_2_original_lax_4c_043_overlay.png'
try:
    img_b64_good = encode_image(good_path)
    payload['messages'][0]['content'][1]['image_url']['url'] = f'data:image/png;base64,{img_b64_good}'
    resp2 = requests.post('http://localhost:11434/v1/chat/completions', json=payload, timeout=300)
    print(resp2.json()['choices'][0]['message']['content'])
except Exception as e:
    print(f'Good case test failed: {e}')
