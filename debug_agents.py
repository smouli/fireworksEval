#!/usr/bin/env python3
"""
Debug script to show what each agent returns from Fireworks LLM.
Shows the actual outputs from each agent step-by-step.
"""

import os
import json
from fireworks_querygpt import FireworksQueryGPT

def load_env_file(env_path: str = ".env") -> None:
    """Load environment variables from .env file."""
    from pathlib import Path
    env_file = Path(env_path)
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    value = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = value

# Load .env
load_env_file()

def debug_agent_outputs():
    """Show what each agent returns from Fireworks."""
    print("=" * 80)
    print("QueryGPT Agent Outputs - What Each Agent Returns from Fireworks")
    print("=" * 80)
    
    if not os.getenv("FIREWORKS_API_KEY"):
        print("Error: FIREWORKS_API_KEY not set in .env file")
        return
    
    question = "How many trips were completed yesterday?"
    print(f"\n📝 User Question: {question}\n")
    
    querygpt = FireworksQueryGPT()
    
    # Step 1: Intent Agent
    print("=" * 80)
    print("STEP 1: INTENT AGENT")
    print("=" * 80)
    print("Purpose: Classify question into a workspace (mobility, customer_analytics, etc.)")
    print("Fireworks Model: Smaller model (intent_model) for fast classification")
    print("Fireworks Feature: Function calling for structured JSON output")
    print("-" * 80)
    
    try:
        intent_result = querygpt.intent_agent(question)
        print("\n✅ Fireworks LLM Returns (via function calling):")
        print(json.dumps(intent_result, indent=2))
        print(f"\n📊 What this means:")
        print(f"   - workspace_id: '{intent_result['workspace_id']}' - Which workspace was selected")
        print(f"   - confidence: {intent_result['confidence']} - How confident the model is (0-1)")
        print(f"   - reasoning: '{intent_result['reasoning']}' - Why this workspace was chosen")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    workspace_id = intent_result["workspace_id"]
    
    # Step 2: Table Agent
    print("\n\n" + "=" * 80)
    print("STEP 2: TABLE AGENT")
    print("=" * 80)
    print("Purpose: Identify which tables are needed to answer the question")
    print("Fireworks Model: Main model (larger) for better reasoning")
    print("Fireworks Feature: Function calling for structured JSON output")
    print("-" * 80)
    
    try:
        table_result = querygpt.table_agent(question, workspace_id)
        print("\n✅ Fireworks LLM Returns (via function calling):")
        print(json.dumps(table_result, indent=2))
        print(f"\n📊 What this means:")
        print(f"   - relevant_tables: {table_result['relevant_tables']} - Tables needed for the query")
        print(f"   - reasoning: '{table_result['reasoning']}' - Why these tables were selected")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    relevant_tables = table_result["relevant_tables"]
    
    # Step 3: Column Prune Agent
    print("\n\n" + "=" * 80)
    print("STEP 3: COLUMN PRUNE AGENT")
    print("=" * 80)
    print("Purpose: Identify which columns are needed (reduces token usage)")
    print("Fireworks Model: Main model")
    print("Fireworks Feature: Function calling for structured JSON output")
    print("-" * 80)
    
    workspace = querygpt.workspaces[workspace_id]
    
    for table_name in relevant_tables[:1]:  # Show first table as example
        print(f"\n🔍 Processing table: {table_name}")
        full_schema = workspace["table_schemas"][table_name]
        
        try:
            prune_result = querygpt.column_prune_agent(question, table_name, full_schema)
            print("\n✅ Fireworks LLM Returns (via function calling):")
            print(json.dumps({
                "relevant_columns": prune_result["relevant_columns"],
                "token_savings_pct": prune_result["token_savings_pct"]
            }, indent=2))
            print(f"\n📊 What this means:")
            print(f"   - relevant_columns: {prune_result['relevant_columns']} - Columns needed for query")
            print(f"   - token_savings_pct: {prune_result['token_savings_pct']:.1f}% - Token reduction achieved")
            print(f"   - Original columns: {len(full_schema['columns'])}")
            print(f"   - Pruned columns: {len(prune_result['pruned_schema']['columns'])}")
        except Exception as e:
            print(f"❌ Error: {e}")
            return
    
    # Step 4: SQL Generation Agent
    print("\n\n" + "=" * 80)
    print("STEP 4: SQL GENERATION AGENT")
    print("=" * 80)
    print("Purpose: Generate the final SQL query")
    print("Fireworks Model: Main model")
    print("Fireworks Feature: Function calling for structured SQL output")
    print("Input: Few-shot examples + pruned schemas + user question")
    print("-" * 80)
    
    # Build pruned schemas for all tables
    pruned_schemas = {}
    for table_name in relevant_tables:
        full_schema = workspace["table_schemas"][table_name]
        prune_result = querygpt.column_prune_agent(question, table_name, full_schema)
        pruned_schemas[table_name] = prune_result["pruned_schema"]
    
    sql_samples = workspace["sql_samples"]
    
    try:
        sql_result = querygpt.sql_generation_agent(
            question,
            workspace_id,
            relevant_tables,
            pruned_schemas,
            sql_samples
        )
        print("\n✅ Fireworks LLM Returns (via function calling):")
        print(json.dumps(sql_result, indent=2))
        print(f"\n📊 What this means:")
        print(f"   - sql_query: '{sql_result['sql_query']}' - The generated SQL query")
        print(f"   - explanation: '{sql_result['explanation']}' - How the query works")
        print(f"   - tables_used: {sql_result['tables_used']} - Tables used in the query")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Final Result
    print("\n\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)
    print("Complete pipeline output:")
    print("-" * 80)
    
    final_result = querygpt.generate_sql(question)
    print(json.dumps({
        "question": final_result["question"],
        "workspace": final_result["workspace"],
        "tables": final_result["tables"],
        "sql": final_result["sql"],
        "explanation": final_result["explanation"],
        "intent_confidence": final_result["intent_confidence"],
        "total_latency_ms": final_result["total_latency_ms"]
    }, indent=2))
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Each agent uses Fireworks LLM with function calling:
1. Intent Agent → Returns: workspace_id, confidence, reasoning
2. Table Agent → Returns: relevant_tables, reasoning  
3. Column Prune Agent → Returns: relevant_columns, pruned_schema, token_savings_pct
4. SQL Generation Agent → Returns: sql_query, explanation, tables_used

All outputs are structured JSON via function calling - no parsing needed!
""")

if __name__ == "__main__":
    debug_agent_outputs()

