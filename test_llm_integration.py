
import os
from cardiac_agent_postproc.settings import LLMSettings
from cardiac_agent_postproc.api_client import OpenAICompatClient

def test_integration():
    print("--- Testing LLM Integration (Env + Client) ---")
    
    # 1. Test Settings Loading
    try:
        settings = LLMSettings()
        print(f"Settings Loaded:")
        print(f"  Enabled: {settings.llm_enabled}")
        print(f"  Base URL: {settings.openai_base_url}")
        print(f"  Model: {settings.openai_model}")
        print(f"  API Key Present: {'Yes' if settings.openai_api_key and settings.openai_api_key != 'ollama' else 'No/Default'}")
        
        if not settings.llm_enabled:
            print("WARNING: LLM_ENABLED is False in settings!")
            
    except Exception as e:
        print(f"FAILED to load settings: {e}")
        return

    # 2. Test Client Connection
    try:
        client = OpenAICompatClient(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            model=settings.openai_model
        )
        
        print(f"\nSending Chat Request to {client.model}...")
        res = client.chat_json(
            system="You are a test bot. Return JSON: {'status': 'ok'}",
            user="Ping",
            max_tokens=50
        )
        print("Response:", res)
        
        if res and isinstance(res, dict):
             print("\nSUCCESS: LLM Connection Verified.")
        else:
             print("\nFAILURE: Invalid response format.")
             
    except Exception as e:
        print(f"\nCRITICAL FAILURE during call: {e}")

if __name__ == "__main__":
    test_integration()
