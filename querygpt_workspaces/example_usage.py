#!/usr/bin/env python3
"""
Example usage of QueryGPT-style workspace system.
Demonstrates how to use workspaces for text-to-SQL generation.
"""

import json
from pathlib import Path


def load_workspaces():
    """Load workspace definitions."""
    with open("querygpt_workspaces/workspaces.json") as f:
        return json.load(f)


def load_intent_mapping():
    """Load intent mapping."""
    with open("querygpt_workspaces/intent_mapping.json") as f:
        return json.load(f)


def infer_intent(question: str, intent_mapping: dict) -> str:
    """
    Simple intent inference (in production, this would use an LLM).
    This is a keyword-based example.
    """
    question_lower = question.lower()
    
    for intent_id, intent_info in intent_mapping["intent_categories"].items():
        keywords = intent_info["keywords"]
        if any(keyword in question_lower for keyword in keywords):
            return intent_id
    
    return "mobility"  # Default


def get_relevant_tables(question: str, workspace: dict) -> list:
    """
    Simple table selection (in production, this would use an LLM).
    Returns all workspace tables for this example.
    """
    return workspace["tables"]


def get_sql_samples(workspace: dict, limit: int = 5) -> list:
    """Get SQL samples for few-shot learning."""
    return workspace["sql_samples"][:limit]


def get_pruned_schema(table_name: str, question: str, workspace: dict) -> dict:
    """
    Simple column pruning example (in production, this would use an LLM).
    Returns full schema for this example, but in production would prune.
    """
    return workspace["table_schemas"].get(table_name, {})


def demonstrate_querygpt_flow(question: str):
    """Demonstrate the QueryGPT flow for a user question."""
    print("=" * 80)
    print(f"User Question: {question}")
    print("=" * 80)
    
    # Load data
    workspaces = load_workspaces()
    intent_mapping = load_intent_mapping()
    
    # Step 1: Intent Agent
    print("\n[Step 1] Intent Agent")
    print("-" * 80)
    intent = infer_intent(question, intent_mapping)
    workspace = workspaces[intent]
    print(f"Detected Intent: {intent}")
    print(f"Workspace: {workspace['name']}")
    print(f"Description: {workspace['description']}")
    
    # Step 2: Table Agent
    print("\n[Step 2] Table Agent")
    print("-" * 80)
    relevant_tables = get_relevant_tables(question, workspace)
    print(f"Relevant Tables: {', '.join(relevant_tables)}")
    
    # Step 3: Column Prune Agent (simplified)
    print("\n[Step 3] Column Prune Agent")
    print("-" * 80)
    for table_name in relevant_tables[:3]:  # Show first 3 tables
        schema = get_pruned_schema(table_name, question, workspace)
        if schema:
            columns = [col["name"] for col in schema["columns"]]
            print(f"{table_name}: {len(columns)} columns")
            print(f"  Sample columns: {', '.join(columns[:5])}...")
    
    # Step 4: SQL Samples for Few-Shot Learning
    print("\n[Step 4] SQL Samples (Few-Shot Examples)")
    print("-" * 80)
    samples = get_sql_samples(workspace, limit=3)
    for i, sample in enumerate(samples, 1):
        print(f"\nExample {i}:")
        print(f"  Q: {sample['question']}")
        print(f"  SQL: {sample['sql']}")
        print(f"  Description: {sample['description']}")
    
    # Step 5: Generate SQL (would use LLM here)
    print("\n[Step 5] SQL Generation")
    print("-" * 80)
    print("(In production, this would use an LLM with:")
    print("  - Few-shot examples from workspace")
    print("  - Pruned table schemas")
    print("  - User question")
    print("  - Business domain instructions)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Example questions
    example_questions = [
        "How many trips were completed by Teslas in Seattle yesterday?",
        "What are the top 5 customers by total spending?",
        "Which vehicle type generates the most revenue?",
        "How many trips used the WELCOME10 promotion?",
    ]
    
    print("QueryGPT-Style Workspace System - Example Usage\n")
    
    for question in example_questions:
        demonstrate_querygpt_flow(question)
        print("\n")

