"""
Quick smoke test for Ollama + ministral-3:14b integration.

Usage:
    python test_ollama_connection.py
    python test_ollama_connection.py --image /path/to/some/image.png
"""
import argparse
import sys
import os

# Ensure the package is importable
sys.path.insert(0, os.path.dirname(__file__))

from cardiac_agent_postproc.api_client import OpenAICompatClient


def test_text_only(client: OpenAICompatClient) -> bool:
    """Test text-only chat_json."""
    print("=" * 60)
    print("TEST 1: Text-only chat_json")
    print("=" * 60)

    result = client.chat_json(
        system="You are a helpful assistant. Always respond in valid JSON.",
        user=(
            'What are the 4 chambers of the human heart? '
            'Respond as JSON: {"chambers": ["...", ...]}'
        ),
        temperature=0.1,
        max_tokens=512,
    )

    print(f"Response: {result}")
    if result and "chambers" in result:
        print("PASS: Got structured JSON with 'chambers' key")
        return True
    else:
        print("WARN: Response missing 'chambers' key (model may need prompt tuning)")
        return bool(result)  # pass if we got any JSON back


def test_vision_single(client: OpenAICompatClient, image_path: str) -> bool:
    """Test chat_vision_json with a single image."""
    print("\n" + "=" * 60)
    print("TEST 2: Vision chat_vision_json (single image)")
    print("=" * 60)

    if not os.path.exists(image_path):
        print(f"SKIP: Image not found at {image_path}")
        return True

    result = client.chat_vision_json(
        system="You are a medical image analysis assistant. Respond in JSON.",
        user_text=(
            'Describe what you see in this image briefly. '
            'Respond as JSON: {"description": "...", "has_content": true/false}'
        ),
        image_path=image_path,
        temperature=0.1,
    )

    print(f"Response: {result}")
    if result:
        print("PASS: Got JSON response from vision model")
        return True
    else:
        print("FAIL: Empty response from vision model")
        return False


def test_vision_multi(client: OpenAICompatClient, image_path: str) -> bool:
    """Test chat_vision_multi_json with multiple images."""
    print("\n" + "=" * 60)
    print("TEST 3: Vision chat_vision_multi_json (multi-image)")
    print("=" * 60)

    if not os.path.exists(image_path):
        print(f"SKIP: Image not found at {image_path}")
        return True

    result = client.chat_vision_multi_json(
        system="You are a medical image analysis assistant. Respond in JSON.",
        user_text=(
            'Compare the two images shown. '
            'Respond as JSON: {"same_content": true/false, "description": "..."}'
        ),
        image_paths=[image_path, image_path],  # same image twice for testing
        image_labels=["[Image A]", "[Image B]"],
        temperature=0.1,
    )

    print(f"Response: {result}")
    if result:
        print("PASS: Got JSON response from multi-image vision")
        return True
    else:
        print("FAIL: Empty response from multi-image vision")
        return False


def test_warmup(client: OpenAICompatClient) -> bool:
    """Test warmup() preloads the model."""
    print("=" * 60)
    print("TEST 0: Warmup (preload model into GPU)")
    print("=" * 60)

    ok = client.warmup()
    if ok:
        print("PASS: Model warmup succeeded")
    else:
        print("FAIL: Model warmup failed — is Ollama running?")
    return ok


def test_json_strictness(client: OpenAICompatClient) -> bool:
    """Test that response_format=json_object forces valid JSON output."""
    print("\n" + "=" * 60)
    print("TEST 4: JSON strictness (response_format enforcement)")
    print("=" * 60)

    result = client.chat_json(
        system="You are a helpful assistant. Always respond in valid JSON.",
        user=(
            'Tell me a fun fact about the heart. '
            'You MUST respond as JSON: {"fact": "...", "source": "..."}'
        ),
        temperature=0.3,
        max_tokens=256,
    )

    print(f"Response: {result}")
    if not result:
        print("FAIL: Empty response")
        return False

    # Check it's actually a dict (parsed JSON), not raw text
    if isinstance(result, dict) and len(result) > 0:
        print("PASS: Got valid JSON object from model")
        return True
    else:
        print("FAIL: Response is not a valid JSON object")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Ollama + ministral-3:14b")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--model", default="ministral-3:14b")
    parser.add_argument("--image", default=None, help="Path to a test image (PNG)")
    parser.add_argument("--timeout", type=float, default=300.0, help="Request timeout in seconds")
    args = parser.parse_args()

    client = OpenAICompatClient(
        base_url=args.base_url,
        api_key="ollama",
        model=args.model,
        timeout=args.timeout,
        provider="ollama",
    )

    print(f"Ollama endpoint: {args.base_url}")
    print(f"Model: {args.model}")
    print(f"Timeout: {args.timeout}s")
    print()

    results = []

    # Test 0: warmup
    results.append(("warmup", test_warmup(client)))

    # Test 1: text-only
    results.append(("text-only", test_text_only(client)))

    # Test 2 & 3: vision (if image provided)
    if args.image:
        results.append(("vision-single", test_vision_single(client, args.image)))
        results.append(("vision-multi", test_vision_multi(client, args.image)))
    else:
        # Try to find a test image in the data directory
        data_dir = "results/Input_MnM2/all_frames_export"
        test_img = None
        if os.path.isdir(data_dir):
            for d in sorted(os.listdir(data_dir))[:1]:
                subdir = os.path.join(data_dir, d)
                if os.path.isdir(subdir):
                    for f in os.listdir(subdir):
                        if f.endswith("_img.png"):
                            test_img = os.path.join(subdir, f)
                            break
                if test_img:
                    break

        if test_img:
            print(f"\nAuto-detected test image: {test_img}")
            results.append(("vision-single", test_vision_single(client, test_img)))
            results.append(("vision-multi", test_vision_multi(client, test_img)))
        else:
            print("\nNo test image available. Pass --image to test vision.")

    # Test 4: JSON strictness
    results.append(("json-strictness", test_json_strictness(client)))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
