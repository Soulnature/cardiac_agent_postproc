
import os
import sys
import yaml
import json
from cardiac_agent_postproc.api_client import OpenAICompatClient

# Mock config
cfg = {
    "provider": "ollama",
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama",
    "model": "ministral-3:14b",
    "temperature": 0.1,
    "max_tokens": 1000
}

def test_vlm():
    print(f"Testing VLM with config: {cfg}")
    client = OpenAICompatClient(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        model=cfg["model"],
        provider=cfg["provider"]
    )
    
    # Create a dummy image or use an existing one
    # Let's look for a png in the current dir or batch_eval_results
    img_path = "test_image.png"
    
    # Create a simple red square if not exists
    if not os.path.exists(img_path):
        import cv2
        import numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:] = (0, 0, 255) # Red
        cv2.imwrite(img_path, img)
        print(f"Created dummy image at {img_path}")
        
    system_prompt = "You are a vision assistant. Describe the image."
    user_prompt = "What color is this image? Respond in JSON: {'color': '...'}"
    
    print("Sending request...")
    try:
        response = client.chat_vision_json(system_prompt, user_prompt, img_path)
        print("Response received:")
        print(json.dumps(response, indent=2))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_vlm()
