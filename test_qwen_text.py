import json
from cardiac_agent_postproc.api_client import OpenAICompatClient

def test():
    client = OpenAICompatClient(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="qwen3-vl:8b",
        provider="ollama"
    )
    
    prompt = "Hello, are you working? Respond with JSON: {'status': 'ok'}"
    
    print("Testing chat_json...")
    res = client.chat_json(
        system="You are a helpful assistant.",
        user=prompt
    )
    print(f"Result: {json.dumps(res, indent=2)}")

if __name__ == "__main__":
    test()
