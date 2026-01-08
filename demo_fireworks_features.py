#!/usr/bin/env python3
"""
Complete demo of Fireworks features: Prompt Caching and PII Protection.
"""

import os
from demo_prompt_caching import PromptCachingDemo
from demo_pii_protection import PIIProtectionDemo

def main():
    if not os.getenv("FIREWORKS_API_KEY"):
        print("Error: FIREWORKS_API_KEY not set")
        print("Set it with: export FIREWORKS_API_KEY='your-key'")
        exit(1)
    
    print("\n" + "=" * 80)
    print("Fireworks Features Demo")
    print("=" * 80)
    
    # Demo 1: Prompt Caching
    print("\n[DEMO 1] Prompt Caching Optimization")
    print("-" * 80)
    caching_demo = PromptCachingDemo()
    caching_demo.demo_caching_benefits()
    
    # Demo 2: PII Protection
    print("\n\n[DEMO 2] PII Protection & Compliance")
    print("-" * 80)
    pii_demo = PIIProtectionDemo()
    pii_demo.demo_pii_protection()
    
    print("\n" + "=" * 80)
    print("Summary:")
    print("  ✓ Prompt Caching: Up to 80% faster TTFT for repeated prompts")
    print("  ✓ PII Protection: Zero data retention, encryption, compliance")
    print("  ✓ Both features enabled by default in Fireworks")
    print("=" * 80)

if __name__ == "__main__":
    main()

