#!/usr/bin/env python3
"""
Compare QueryGPT agents with and without prompt caching optimization.
Shows performance difference when prompts are structured for caching.
"""

import os
import time
import json
from typing import Dict, List, Any
from pathlib import Path
from fireworks_querygpt import FireworksQueryGPT, AgentMetrics
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
                    # Remove quotes if present
                    value = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = value


# Load .env file if it exists
load_env_file()


class OptimizedFireworksQueryGPT(FireworksQueryGPT):
    """
    Optimized version that structures prompts for maximum cache hits.
    Static content (system prompts, schemas) goes FIRST.
    Variable content (user questions) goes LAST.
    """
    
    def intent_agent(self, question: str) -> Dict[str, Any]:
        """Optimized intent agent with caching-friendly prompt structure."""
        start_time = time.time()
        
        intent_function = {
            "type": "function",
            "function": {
                "name": "classify_intent",
                "description": "Classify the user's question into a business domain workspace",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workspace_id": {
                            "type": "string",
                            "enum": list(self.workspaces.keys()),
                            "description": "The workspace ID that best matches the question"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence score between 0 and 1"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief explanation of why this workspace was chosen"
                        }
                    },
                    "required": ["workspace_id", "confidence", "reasoning"]
                }
            }
        }
        
        # OPTIMIZED: Static content FIRST (will be cached)
        workspace_descriptions = "\n".join([
            f"- {ws_id}: {ws['name']} - {ws['description']}"
            for ws_id, ws in self.workspaces.items()
        ])
        
        static_prompt = f"""You are an intent classification agent for a text-to-SQL system.

Available workspaces:
{workspace_descriptions}

Classify questions into the most appropriate workspace. Consider keywords, domain, and query intent.

"""
        
        # Variable content LAST (changes per request)
        variable_prompt = f"User question: {question}"
        
        # Combined: static first, variable last (optimal for caching)
        prompt = static_prompt + variable_prompt

        try:
            response = self.client.chat.completions.create(
                model=self.intent_model,
                messages=[{"role": "user", "content": prompt}],
                tools=[intent_function],
                tool_choice={"type": "function", "function": {"name": "classify_intent"}},
                temperature=0.1,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            function_call = response.choices[0].message.tool_calls[0]
            result = json.loads(function_call.function.arguments)
            
            self.metrics.append(AgentMetrics(
                agent_name="intent_agent_optimized",
                latency_ms=latency_ms,
                tokens_used=response.usage.total_tokens,
                function_called=True,
                accuracy=result.get("confidence")
            ))
            
            return result
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.append(AgentMetrics(
                agent_name="intent_agent_optimized",
                latency_ms=latency_ms,
                tokens_used=0,
                function_called=False,
                error=str(e)
            ))
            raise
    
    def sql_generation_agent(
        self,
        question: str,
        workspace_id: str,
        relevant_tables: List[str],
        pruned_schemas: Dict[str, Dict],
        sql_samples: List[Dict]
    ) -> Dict[str, Any]:
        """Optimized SQL generation agent with caching-friendly prompt structure."""
        start_time = time.time()
        workspace = self.workspaces[workspace_id]
        
        sql_function = {
            "type": "function",
            "function": {
                "name": "generate_sql",
                "description": "Generate a SQL query to answer the user's question",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql_query": {
                            "type": "string",
                            "description": "The complete SQL query"
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Brief explanation of how the query works"
                        },
                        "tables_used": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of tables used in the query"
                        }
                    },
                    "required": ["sql_query", "explanation", "tables_used"]
                }
            }
        }
        
        # OPTIMIZED: Static content FIRST (will be cached)
        few_shot_examples = "\n\n".join([
            f"Q: {sample['question']}\nSQL: {sample['sql']}"
            for sample in sql_samples[:5]
        ])
        
        schema_context = "\n\n".join([
            f"Table: {table_name}\nColumns: {', '.join([col['name'] for col in schema['columns']])}"
            for table_name, schema in pruned_schemas.items()
        ])
        
        static_prompt = f"""You are a SQL generation agent. Generate a SQLite query to answer the user's question.

Workspace: {workspace['name']}
Domain: {workspace['description']}

Few-shot examples:
{few_shot_examples}

Table schemas (pruned):
{schema_context}

Generate a valid SQLite query. Ensure:
- Correct table and column names
- Proper JOIN syntax
- Valid WHERE/GROUP BY/ORDER BY clauses
- SQLite-compatible functions

"""
        
        # Variable content LAST (changes per request)
        variable_prompt = f"User question: {question}"
        
        # Combined: static first, variable last (optimal for caching)
        prompt = static_prompt + variable_prompt

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=[sql_function],
                tool_choice={"type": "function", "function": {"name": "generate_sql"}},
                temperature=0.3,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            function_call = response.choices[0].message.tool_calls[0]
            result = json.loads(function_call.function.arguments)
            
            self.metrics.append(AgentMetrics(
                agent_name="sql_generation_agent_optimized",
                latency_ms=latency_ms,
                tokens_used=response.usage.total_tokens,
                function_called=True
            ))
            
            return result
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.append(AgentMetrics(
                agent_name="sql_generation_agent_optimized",
                latency_ms=latency_ms,
                tokens_used=0,
                function_called=False,
                error=str(e)
            ))
            raise


def compare_agents():
    """Compare standard vs optimized agents."""
    print("=" * 80)
    print("QueryGPT Agent Comparison: Standard vs Optimized (Prompt Caching)")
    print("=" * 80)
    
    if not os.getenv("FIREWORKS_API_KEY"):
        print("Error: FIREWORKS_API_KEY not set in .env file")
        print("Create a .env file with: FIREWORKS_API_KEY=your-key-here")
        return
    
    # Test questions
    test_questions = [
        "How many trips were completed yesterday?",
        "What are the top 5 drivers by rating?",
        "Which city has the most trips?",
        "Show all trips in Seattle",
        "Count trips by driver",
    ]
    
    print(f"\nTesting with {len(test_questions)} questions...")
    print("Note: First request is 'cold cache', subsequent requests benefit from caching\n")
    
    # Test Standard Implementation
    print("\n" + "=" * 80)
    print("STANDARD IMPLEMENTATION (Current)")
    print("=" * 80)
    print("Prompt structure: Mixed static/variable content")
    print("-" * 80)
    
    standard_qgpt = FireworksQueryGPT()
    standard_times = []
    
    for i, question in enumerate(test_questions, 1):
        try:
            start = time.time()
            result = standard_qgpt.generate_sql(question)
            elapsed = (time.time() - start) * 1000
            
            standard_times.append(elapsed)
            cache_status = "Cold" if i == 1 else "Warm"
            
            print(f"\n[{i}] {cache_status} Cache - {question[:50]}...")
            print(f"    Latency: {elapsed:.0f}ms")
            if "sql" in result:
                print(f"    SQL: {result['sql'][:60]}...")
        except Exception as e:
            print(f"    Error: {e}")
    
    # Test Optimized Implementation
    print("\n\n" + "=" * 80)
    print("OPTIMIZED IMPLEMENTATION (Caching-Friendly)")
    print("=" * 80)
    print("Prompt structure: Static content FIRST, variable content LAST")
    print("-" * 80)
    
    optimized_qgpt = OptimizedFireworksQueryGPT()
    optimized_times = []
    
    for i, question in enumerate(test_questions, 1):
        try:
            start = time.time()
            result = optimized_qgpt.generate_sql(question)
            elapsed = (time.time() - start) * 1000
            
            optimized_times.append(elapsed)
            cache_status = "Cold" if i == 1 else "Warm"
            
            print(f"\n[{i}] {cache_status} Cache - {question[:50]}...")
            print(f"    Latency: {elapsed:.0f}ms")
            if "sql" in result:
                print(f"    SQL: {result['sql'][:60]}...")
        except Exception as e:
            print(f"    Error: {e}")
    
    # Compare Results
    print("\n\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    
    if standard_times and optimized_times:
        standard_avg = sum(standard_times[1:]) / len(standard_times[1:]) if len(standard_times) > 1 else standard_times[0]
        optimized_avg = sum(optimized_times[1:]) / len(optimized_times[1:]) if len(optimized_times) > 1 else optimized_times[0]
        
        print(f"\nStandard (warm cache avg): {standard_avg:.0f}ms")
        print(f"Optimized (warm cache avg): {optimized_avg:.0f}ms")
        
        if standard_avg > 0:
            improvement = ((standard_avg - optimized_avg) / standard_avg) * 100
            print(f"\nImprovement: {improvement:.1f}% faster with optimized prompts")
        
        print(f"\nCold cache (first request):")
        print(f"  Standard: {standard_times[0]:.0f}ms")
        print(f"  Optimized: {optimized_times[0]:.0f}ms")
        
        print(f"\nWarm cache (subsequent requests):")
        if len(standard_times) > 1:
            print(f"  Standard avg: {sum(standard_times[1:]) / len(standard_times[1:]):.0f}ms")
        if len(optimized_times) > 1:
            print(f"  Optimized avg: {sum(optimized_times[1:]) / len(optimized_times[1:]):.0f}ms")
    
    # Show Metrics
    print("\n" + "=" * 80)
    print("AGENT METRICS")
    print("=" * 80)
    
    print("\nStandard Implementation:")
    standard_metrics = standard_qgpt.get_metrics_summary()
    for agent, metrics in standard_metrics.items():
        print(f"  {agent}:")
        print(f"    Avg Latency: {metrics['avg_latency_ms']:.0f}ms")
        print(f"    Total Tokens: {metrics['total_tokens']}")
    
    print("\nOptimized Implementation:")
    optimized_metrics = optimized_qgpt.get_metrics_summary()
    for agent, metrics in optimized_metrics.items():
        print(f"  {agent}:")
        print(f"    Avg Latency: {metrics['avg_latency_ms']:.0f}ms")
        print(f"    Total Tokens: {metrics['total_tokens']}")
    
    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("✓ Prompt caching works automatically in Fireworks")
    print("✓ Structuring prompts (static first, variable last) maximizes cache hits")
    print("✓ Can achieve up to 80% reduction in Time To First Token (TTFT)")
    print("✓ Especially beneficial for repeated queries with same schema/examples")
    print("=" * 80)


if __name__ == "__main__":
    compare_agents()

