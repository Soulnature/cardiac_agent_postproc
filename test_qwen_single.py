import json
import cv2
import numpy as np
import base64
from openai import OpenAI

def test():
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )
    
    # Create a dummy image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (80, 80), (0, 255, 255), -1)
    cv2.imwrite("test_vlm_v2.png", img)
    
    with open("test_vlm_v2.png", "rb") as f:
        b64_img = base64.b64encode(f.read()).decode("utf-8")
    
    print("Testing single image multimodal...")
    try:
        response = client.chat.completions.create(
            model="qwen3-vl:8b",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                ]}
            ],
            max_tokens=100
        )
        print(f"Result: {response.choices[0].message.content}")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    test()
