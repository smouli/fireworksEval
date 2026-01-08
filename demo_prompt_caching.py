#!/usr/bin/env python3
"""
Demonstrate Fireworks Prompt Caching optimization.
Shows how structuring prompts with static content at the beginning maximizes cache hits.
"""

import os
import time
from openai import OpenAI

class PromptCachingDemo:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("FIREWORKS_API_KEY"),
            base_url="https://api.fireworks.ai/inference/v1"
        )
        self.model = os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/firefunction-v2")
    
    def measure_ttft(self, prompt: str) -> float:
        """Measure Time To First Token (TTFT)."""
        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                return (time.time() - start) * 1000
        return 0
    
    def demo_caching_benefits(self):
        """Demonstrate prompt caching by showing faster responses for repeated prompts."""
        print("=" * 80)
        print("Fireworks Prompt Caching Demo")
        print("=" * 80)
        
        static_prompt = """You are a SQL generation assistant. Generate SQLite queries.

Database schema:
- trips (trip_id, driver_id, customer_id, total_amount, pickup_time)
- drivers (driver_id, first_name, last_name, rating)
- cities (city_id, city_name, state)

"""
        
        questions = [
            "How many trips were completed yesterday?",
            "What are the top 5 drivers by rating?",
            "Which city has the most trips?",
        ]
        
        print("\n1. First request (cold cache):")
        prompt1 = static_prompt + f"Question: {questions[0]}"
        ttft1 = self.measure_ttft(prompt1)
        print(f"   TTFT: {ttft1:.0f}ms")
        
        print("\n2. Subsequent requests (warm cache):")
        ttft_times = []
        for i, question in enumerate(questions[1:], 2):
            prompt = static_prompt + f"Question: {question}"
            ttft = self.measure_ttft(prompt)
            ttft_times.append(ttft)
            print(f"   Request {i}: {ttft:.0f}ms")
        
        if ttft_times:
            avg_ttft = sum(ttft_times) / len(ttft_times)
            improvement = ((ttft1 - avg_ttft) / ttft1) * 100 if ttft1 > 0 else 0
            print(f"\n   Average TTFT (cached): {avg_ttft:.0f}ms")
            print(f"   Improvement: {improvement:.1f}% faster")
            print(f"   Cache benefit: Up to 80% reduction in TTFT")
        
        print("\n" + "=" * 80)
        print("Key Takeaway:")
        print("  - Static content (schemas, instructions) at the BEGINNING")
        print("  - Variable content (user questions) at the END")
        print("  - Fireworks automatically caches common prefixes")

if __name__ == "__main__":
    if not os.getenv("FIREWORKS_API_KEY"):
        print("Error: FIREWORKS_API_KEY not set")
        exit(1)
    
    demo = PromptCachingDemo()
    demo.demo_caching_benefits()

