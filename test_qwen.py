import os
import sys
import json
import cv2
import numpy as np
from cardiac_agent_postproc.api_client import OpenAICompatClient

def test():
    client = OpenAICompatClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="qwen3-vl:8b",
        provider="ollama"
    )
    
    # Create a dummy image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (80, 80), (0, 255, 255), -1)
    cv2.imwrite("test_vlm.png", img)
    
    prompt = "Evaluate the image quality. Return JSON: {'score': 0-100, 'reason': '...'}"
    
    print("Testing chat_vision_json...")
    res = client.chat_vision_json(
        system="You are a judge.",
        user_text=prompt,
        image_path="test_vlm.png"
    )
    print(f"Result: {json.dumps(res, indent=2)}")

if __name__ == "__main__":
    test()
