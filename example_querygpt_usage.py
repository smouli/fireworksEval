#!/usr/bin/env python3
"""
Example usage of Fireworks QueryGPT implementation.
"""

import os
from fireworks_querygpt import FireworksQueryGPT


def main():
    # Check for API key
    if not os.getenv("FIREWORKS_API_KEY"):
        print("Error: FIREWORKS_API_KEY environment variable not set")
        print("Set it with: export FIREWORKS_API_KEY='your-key-here'")
        return
    
    # Initialize QueryGPT
    print("Initializing Fireworks QueryGPT...")
    querygpt = FireworksQueryGPT()
    
    # Example questions
    questions = [
        "How many trips were completed by Teslas in Seattle yesterday?",
        "What are the top 5 customers by total spending?",
        "Which vehicle type generates the most revenue?",
        "How many trips used the WELCOME10 promotion?",
    ]
    
    print("\n" + "=" * 80)
    print("QueryGPT Example Usage")
    print("=" * 80)
    
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] Question: {question}")
        print("-" * 80)
        
        try:
            result = querygpt.generate_sql(question)
            
            if "error" in result:
                print(f"✗ Error: {result['error']}")
                continue
            
            print(f"✓ Workspace: {result['workspace']}")
            print(f"✓ Tables: {', '.join(result['tables'])}")
            print(f"✓ Intent Confidence: {result['intent_confidence']:.2f}")
            print(f"✓ Latency: {result['total_latency_ms']:.0f}ms")
            print(f"\nGenerated SQL:")
            print(result['sql'])
            print(f"\nExplanation: {result['explanation']}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Show metrics summary
    print("\n" + "=" * 80)
    print("Agent Metrics Summary")
    print("=" * 80)
    
    import json
    metrics = querygpt.get_metrics_summary()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

