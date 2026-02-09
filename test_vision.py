
import os
import yaml
import numpy as np
import cv2
from cardiac_agent_postproc.vlm_guardrail import VisionGuardrail

def test():
    # Load config
    with open("config/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    print(f"Testing Vision Model: {cfg['vision_guardrail']['model']}")
    
    # Load env manually for test
    from dotenv import load_dotenv
    load_dotenv()
    
    print(f"API KEY Present: {'yes' if os.getenv('NVIDIA_API_KEY') else 'no'}")
    print(f"BASE URL: {os.getenv('NVIDIA_BASE_URL')}")

    # Init Guardrail
    guard = VisionGuardrail(cfg)
    
    # Direct Requests Test
    import requests, base64
    
    img_path = "results/inputs/335_original_lax_4c_019_img.png"
    if not os.path.exists(img_path):
        for root, _, files in os.walk("results/inputs"):
             for f in files:
                if f.endswith("_img.png"):
                     img_path = os.path.join(root, f)
                     break
             if img_path: break
    
    print(f"Using image: {img_path}")
    
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')
        
    invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
      "Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}",
      "Accept": "application/json"
    }
    
    payload = {
      "model": cfg['vision_guardrail']['model'],
      "messages": [
        {"role":"user", "content":[
            {"type":"text", "text":"Describe this image carefully."},
            {"type":"image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        ]}
      ],
      "max_tokens": 1024,
      "temperature": 0.2,
      "stream": False
    }
    
    print("Sending DIRECT POST request...")
    try:
        response = requests.post(invoke_url, headers=headers, json=payload, timeout=120)
        print(f"Status: {response.status_code}")
        print("Response:", response.text[:500])
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test()
