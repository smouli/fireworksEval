#!/usr/bin/env python3
"""
Find available Fireworks models for your account.
"""

import os
from pathlib import Path
from openai import OpenAI

def load_env_file(env_path: str = ".env") -> None:
    """Load environment variables from .env file."""
    env_file = Path(env_path)
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    value = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = value

load_env_file()

api_key = os.getenv("FIREWORKS_API_KEY")
if not api_key:
    print("Error: FIREWORKS_API_KEY not found in .env file")
    exit(1)

client = OpenAI(
    api_key=api_key,
    base_url="https://api.fireworks.ai/inference/v1"
)

print("=" * 80)
print("Finding Available Fireworks Models")
print("=" * 80)
print("\nTrying common model names...\n")

# Common Fireworks model patterns
test_models = [
    "accounts/fireworks/models/firefunction-v2",
    "accounts/fireworks/models/llama-v3-70b-instruct",
    "accounts/fireworks/models/llama-v3-8b-instruct",
    "accounts/fireworks/models/llama-v3p3-70b-instruct",
    "accounts/fireworks/models/llama-v3p3-8b-instruct",
    "fireworks/models/firefunction-v2",
    "fireworks/models/llama-v3-70b-instruct",
    "fireworks/models/llama-v3-8b-instruct",
    "firefunction-v2",
    "llama-v3-70b-instruct",
    "llama-v3-8b-instruct",
]

available_models = []

for model in test_models:
    try:
        # Try a simple completion to see if model exists
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        available_models.append(model)
        print(f"✓ {model} - Available")
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "NOT_FOUND" in error_msg:
            print(f"✗ {model} - Not found")
        else:
            # If it's not a 404, the model might exist but have other issues
            print(f"? {model} - Error (might exist): {error_msg[:50]}")

print("\n" + "=" * 80)
if available_models:
    print("AVAILABLE MODELS:")
    print("=" * 80)
    for model in available_models:
        print(f"  {model}")
    print("\nAdd these to your .env file:")
    print(f"FIREWORKS_MODEL={available_models[0]}")
    if len(available_models) > 1:
        print(f"FIREWORKS_INTENT_MODEL={available_models[-1] if '8b' in available_models[-1] else available_models[0]}")
else:
    print("No models found with common names.")
    print("\nPlease check your Fireworks dashboard:")
    print("  https://app.fireworks.ai/models")
    print("\nThen update your .env file with:")
    print("  FIREWORKS_MODEL=your-model-name-here")
    print("  FIREWORKS_INTENT_MODEL=your-intent-model-name-here")
print("=" * 80)

