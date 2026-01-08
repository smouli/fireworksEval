#!/usr/bin/env python3
"""
Test different Fireworks model names to find the correct format.
"""

import os
from openai import OpenAI

api_key = os.getenv("FIREWORKS_API_KEY", "fw_3ZahrBVt66YEAEbfUAAnXwbN")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.fireworks.ai/inference/v1"
)

# Common Fireworks model name formats to try
model_names = [
    "accounts/fireworks/models/firefunction-v2",
    "accounts/fireworks/models/llama-v3-70b-instruct",
    "accounts/fireworks/models/llama-v3-8b-instruct",
    "fireworks/models/firefunction-v2",
    "fireworks/models/llama-v3-70b-instruct",
    "fireworks/models/llama-v3-8b-instruct",
    "firefunction-v2",
    "llama-v3-70b-instruct",
    "llama-v3-8b-instruct",
]

print("Testing Fireworks model names...")
print("=" * 80)

for model in model_names:
    try:
        print(f"\nTrying: {model}")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        print(f"✓ SUCCESS! Model '{model}' works")
        print(f"  Response: {response.choices[0].message.content}")
        break
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "NOT_FOUND" in error_msg:
            print(f"✗ Model not found")
        else:
            print(f"✗ Error: {error_msg[:100]}")

print("\n" + "=" * 80)
print("If none worked, check: https://app.fireworks.ai/models for available models")

