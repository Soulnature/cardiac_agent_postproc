import sys
import os

print("Step 1: Script started", flush=True)

# Ensure the current directory is in the python path
sys.path.append(os.getcwd())

try:
    print("Step 2: Importing settings...", flush=True)
    from cardiac_agent_postproc.settings import LLMSettings
    print("Step 3: Importing api_client...", flush=True)
    from cardiac_agent_postproc.api_client import OpenAICompatClient
    print("Step 4: Imports done", flush=True)
except Exception as e:
    print(f"Import Error: {e}", flush=True)
    sys.exit(1)

def test_connection():
    print("--- Testing LLM Connection ---", flush=True)
    
    # 1. Load Settings
    try:
        settings = LLMSettings()
        print(f"✅ Settings loaded from .env", flush=True)
        print(f"   Base URL: {settings.openai_base_url}", flush=True)
        print(f"   Model:    {settings.openai_model}", flush=True)
        
    except Exception as e:
        print(f"❌ Failed to load settings: {e}", flush=True)
        return

    # 2. Initialize Client
    try:
        client = OpenAICompatClient(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            model=settings.openai_model
        )
        print("✅ Client initialized", flush=True)
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}", flush=True)
        return

    # 3. Send Request
    print("\nSending test request (JSON mode + Thinking)...", flush=True)
    system_prompt = "You are a helpful assistant. Output JSON."
    user_prompt = "Return a JSON object with a key 'message' saying 'Hello from NVIDIA LLM!'."

    try:
        response = client.chat_json(system_prompt, user_prompt)
        print("\n--- Response Received ---", flush=True)
        print(response, flush=True)
        
    except Exception as e:
        print(f"\n❌ API Request Failed: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_connection()
