#!/usr/bin/env python3
"""
Comprehensive evaluation of QueryGPT agents against golden dataset.
Supports both OpenAI and Fireworks backends.
Tests each agent separately and compares against ground truth.
"""

import os
import json
import re
import sys
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from openai import OpenAI
from utils import load_db, query_db
from fireworks_querygpt import FireworksQueryGPT, load_env_file


# Load .env file
load_env_file()


def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison (whitespace, case-insensitive)."""
    sql = sql.strip()
    sql = re.sub(r'\s+', ' ', sql)  # Normalize whitespace
    sql = sql.lower()
    # Remove trailing semicolons
    sql = sql.rstrip(';')
    return sql


def sql_similarity(sql1: str, sql2: str) -> float:
    """
    Calculate similarity between two SQL queries.
    Returns a score between 0 and 1.
    """
    norm1 = normalize_sql(sql1)
    norm2 = normalize_sql(sql2)
    
    if norm1 == norm2:
        return 1.0
    
    # Extract key components
    def extract_components(sql: str):
        components = {
            'tables': set(re.findall(r'\bfrom\s+(\w+)\b|\bjoin\s+(\w+)\b', sql, re.IGNORECASE)),
            'select_cols': set(re.findall(r'\bselect\s+(.+?)\s+from', sql, re.IGNORECASE)),
            'where': set(re.findall(r'\bwhere\s+(.+?)(?:\s+group|\s+order|\s+limit|$)', sql, re.IGNORECASE)),
        }
        # Flatten tuples
        components['tables'] = {t[0] if t[0] else t[1] for t in components['tables']}
        components['select_cols'] = {c[0].strip() for c in components['select_cols']}
        components['where'] = {w[0].strip() for w in components['where']}
        return components
    
    comp1 = extract_components(sql1)
    comp2 = extract_components(sql2)
    
    # Calculate overlap for each component
    table_overlap = len(comp1['tables'] & comp2['tables']) / max(len(comp1['tables'] | comp2['tables']), 1)
    select_overlap = len(comp1['select_cols'] & comp2['select_cols']) / max(len(comp1['select_cols'] | comp2['select_cols']), 1)
    where_overlap = len(comp1['where'] & comp2['where']) / max(len(comp1['where'] | comp2['where']), 1) if comp1['where'] or comp2['where'] else 1.0
    
    # Weighted average
    similarity = (table_overlap * 0.4 + select_overlap * 0.4 + where_overlap * 0.2)
    return similarity


def compare_results(actual: List[Dict], expected: List[Dict], tolerance: float = 0.01) -> Tuple[bool, Dict[str, Any]]:
    """
    Compare actual query results with expected results.
    Returns (match, details).
    """
    if len(actual) != len(expected):
        return False, {
            "match": False,
            "row_count_match": False,
            "actual_rows": len(actual),
            "expected_rows": len(expected),
            "column_match": None,
            "value_match": None
        }
    
    if len(actual) == 0:
        return True, {
            "match": True,
            "row_count_match": True,
            "actual_rows": 0,
            "expected_rows": 0,
            "column_match": True,
            "value_match": True
        }
    
    # Check column names
    actual_cols = set(actual[0].keys())
    expected_cols = set(expected[0].keys())
    cols_match = actual_cols == expected_cols
    
    if not cols_match:
        return False, {
            "match": False,
            "row_count_match": True,
            "actual_rows": len(actual),
            "expected_rows": len(expected),
            "column_match": False,
            "actual_columns": list(actual_cols),
            "expected_columns": list(expected_cols),
            "value_match": None
        }
    
    # Check values (convert all to strings for comparison, handle numeric tolerance)
    values_match = True
    mismatches = []
    
    for i, (act_row, exp_row) in enumerate(zip(actual, expected)):
        for col in actual_cols:
            act_val = act_row[col]
            exp_val = exp_row[col]
            
            # Handle numeric comparison with tolerance
            try:
                act_num = float(act_val)
                exp_num = float(exp_val)
                if abs(act_num - exp_num) > tolerance:
                    values_match = False
                    mismatches.append({
                        "row": i,
                        "column": col,
                        "actual": act_val,
                        "expected": exp_val
                    })
            except (ValueError, TypeError):
                # String comparison
                if str(act_val).strip() != str(exp_val).strip():
                    values_match = False
                    mismatches.append({
                        "row": i,
                        "column": col,
                        "actual": act_val,
                        "expected": exp_val
                    })
    
    match = cols_match and values_match
    
    return match, {
        "match": match,
        "row_count_match": True,
        "actual_rows": len(actual),
        "expected_rows": len(expected),
        "column_match": cols_match,
        "value_match": values_match,
        "mismatches": mismatches[:5] if mismatches else []  # Show first 5 mismatches
    }


def extract_tables_from_sql(sql: str) -> List[str]:
    """Extract table names from SQL query."""
    tables = set()
    tables.update(re.findall(r'\bFROM\s+(\w+)', sql, re.IGNORECASE))
    tables.update(re.findall(r'\bJOIN\s+(\w+)', sql, re.IGNORECASE))
    return list(tables)


class AgentEvaluator:
    """Evaluate QueryGPT agents against golden dataset."""
    
    def __init__(self, querygpt: FireworksQueryGPT, db_conn, golden_data: List[Dict]):
        self.querygpt = querygpt
        self.db_conn = db_conn
        self.golden_data = golden_data
        self.results = {
            "intent_agent": [],
            "table_agent": [],
            "column_prune_agent": [],
            "sql_generation_agent": [],
            "overall": []
        }
    
    def evaluate_intent_agent(self, test_case: Dict, expected_workspace: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate Intent Agent: Check if workspace classification is correct."""
        question = test_case["question"]
        
        try:
            intent_result = self.querygpt.intent_agent(question)
            predicted_workspace = intent_result.get("workspace_id", "")
            confidence = intent_result.get("confidence", 0)
            
            # Convert confidence to float if string
            if isinstance(confidence, str):
                try:
                    confidence = float(confidence)
                except ValueError:
                    confidence = 0.0
            
            # Infer expected workspace from SQL or use provided
            if expected_workspace is None:
                expected_tables = extract_tables_from_sql(test_case["sql"])
                # Map tables to workspace (simplified heuristic)
                if any(t in expected_tables for t in ["trips", "drivers", "vehicles"]):
                    expected_workspace = "mobility"
                elif any(t in expected_tables for t in ["customers"]):
                    expected_workspace = "customer_analytics"
                elif any(t in expected_tables for t in ["promotions"]):
                    expected_workspace = "promotions"
                else:
                    expected_workspace = None
            
            correct = (expected_workspace is None) or (predicted_workspace == expected_workspace)
            
            return {
                "question": question,
                "predicted_workspace": predicted_workspace,
                "expected_workspace": expected_workspace,
                "correct": correct,
                "confidence": confidence,
                "error": None
            }
        except Exception as e:
            return {
                "question": question,
                "predicted_workspace": None,
                "expected_workspace": expected_workspace,
                "correct": False,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def evaluate_table_agent(self, test_case: Dict, workspace_id: str) -> Dict[str, Any]:
        """Evaluate Table Agent: Check if correct tables are selected."""
        question = test_case["question"]
        expected_tables = set(extract_tables_from_sql(test_case["sql"]))
        
        try:
            table_result = self.querygpt.table_agent(question, workspace_id)
            predicted_tables = set(table_result.get("relevant_tables", []))
            
            if expected_tables:
                precision = len(predicted_tables & expected_tables) / len(predicted_tables) if predicted_tables else 0.0
                recall = len(predicted_tables & expected_tables) / len(expected_tables) if expected_tables else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            else:
                precision = recall = f1 = 0.0
            
            return {
                "question": question,
                "predicted_tables": list(predicted_tables),
                "expected_tables": list(expected_tables),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "correct": f1 >= 0.9,  # Consider correct if F1 >= 0.9
                "error": None
            }
        except Exception as e:
            return {
                "question": question,
                "predicted_tables": [],
                "expected_tables": list(expected_tables),
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "correct": False,
                "error": str(e)
            }
    
    def evaluate_column_prune_agent(self, test_case: Dict, workspace_id: str, table_name: str) -> Dict[str, Any]:
        """Evaluate Column Prune Agent: Check if relevant columns are kept."""
        question = test_case["question"]
        expected_sql = test_case["sql"]
        
        # Extract expected columns from SQL
        expected_cols = set()
        # SELECT columns
        select_match = re.search(r'SELECT\s+(.+?)\s+FROM', expected_sql, re.IGNORECASE)
        if select_match:
            cols_str = select_match.group(1)
            # Handle DISTINCT, CASE, etc.
            cols = [c.strip() for c in cols_str.split(',')]
            for col in cols:
                # Extract column name (remove aliases, functions)
                col_name = re.sub(r'\s+as\s+\w+$', '', col, flags=re.IGNORECASE).strip()
                col_name = re.sub(r'^\w+\.', '', col_name).strip()
                if col_name:
                    expected_cols.add(col_name)
        
        try:
            workspace = self.querygpt.workspaces[workspace_id]
            full_schema = workspace["table_schemas"][table_name]
            
            prune_result = self.querygpt.column_prune_agent(question, table_name, full_schema)
            relevant_cols = set(prune_result.get("relevant_columns", []))
            
            if expected_cols:
                precision = len(relevant_cols & expected_cols) / len(relevant_cols) if relevant_cols else 0.0
                recall = len(relevant_cols & expected_cols) / len(expected_cols) if expected_cols else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            else:
                precision = recall = f1 = 0.0
            
            token_savings = prune_result.get("token_savings_pct", 0.0)
            
            return {
                "question": question,
                "table": table_name,
                "predicted_columns": list(relevant_cols),
                "expected_columns": list(expected_cols),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "token_savings_pct": token_savings,
                "correct": f1 >= 0.7,  # More lenient for column pruning
                "error": None
            }
        except Exception as e:
            return {
                "question": question,
                "table": table_name,
                "predicted_columns": [],
                "expected_columns": list(expected_cols),
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "token_savings_pct": 0.0,
                "correct": False,
                "error": str(e)
            }
    
    def evaluate_sql_generation_agent(
        self,
        test_case: Dict,
        workspace_id: str,
        relevant_tables: List[str],
        pruned_schemas: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """Evaluate SQL Generation Agent: Check SQL query and results."""
        question = test_case["question"]
        expected_sql = test_case["sql"]
        expected_result = test_case.get("expected_result", [])
        
        try:
            workspace = self.querygpt.workspaces[workspace_id]
            sql_samples = workspace["sql_samples"]
            
            sql_result = self.querygpt.sql_generation_agent(
                question,
                workspace_id,
                relevant_tables,
                pruned_schemas,
                sql_samples
            )
            
            generated_sql = sql_result.get("sql_query", "")
            
            # Compare SQL queries
            sql_sim = sql_similarity(generated_sql, expected_sql)
            
            # Execute and compare results
            result_match = False
            result_details = {}
            
            try:
                actual_results = query_db(self.db_conn, generated_sql, return_as_df=False)
                result_match, result_details = compare_results(actual_results, expected_result)
            except Exception as e:
                result_details = {
                    "match": False,
                    "error": str(e),
                    "sql_executable": False
                }
            
            return {
                "question": question,
                "generated_sql": generated_sql,
                "expected_sql": expected_sql,
                "sql_similarity": sql_sim,
                "sql_exact_match": sql_sim == 1.0,
                "result_match": result_match,
                "result_details": result_details,
                "correct": sql_sim >= 0.9 or result_match,  # Consider correct if SQL similar or results match
                "error": None
            }
        except Exception as e:
            return {
                "question": question,
                "generated_sql": "",
                "expected_sql": expected_sql,
                "sql_similarity": 0.0,
                "sql_exact_match": False,
                "result_match": False,
                "result_details": {"error": str(e)},
                "correct": False,
                "error": str(e)
            }
    
    def evaluate_all(self, max_test_cases: Optional[int] = None, verbose: bool = True):
        """Evaluate all agents on golden dataset."""
        test_cases = self.golden_data[:max_test_cases] if max_test_cases else self.golden_data
        
        print(f"\n{'='*80}")
        print(f"Evaluating {len(test_cases)} test cases")
        print(f"{'='*80}\n")
        
        for i, test_case in enumerate(test_cases):
            if verbose:
                print(f"\n[{i+1}/{len(test_cases)}] {test_case['question']}")
            
            try:
                # Step 1: Intent Agent
                intent_eval = self.evaluate_intent_agent(test_case)
                self.results["intent_agent"].append(intent_eval)
                if verbose:
                    print(f"  Intent: {intent_eval['predicted_workspace']} (expected: {intent_eval['expected_workspace']}, correct: {intent_eval['correct']})")
                
                workspace_id = intent_eval.get("predicted_workspace", "mobility")
                
                # Step 2: Table Agent
                table_eval = self.evaluate_table_agent(test_case, workspace_id)
                self.results["table_agent"].append(table_eval)
                if verbose:
                    print(f"  Tables: {table_eval['predicted_tables']} (expected: {table_eval['expected_tables']}, F1: {table_eval['f1']:.2f})")
                
                # Step 3: Column Prune Agent (for first table)
                relevant_tables = table_eval.get("predicted_tables", [])
                pruned_schemas = {}
                if relevant_tables:
                    table_name = relevant_tables[0]  # Evaluate on first table
                    prune_eval = self.evaluate_column_prune_agent(test_case, workspace_id, table_name)
                    self.results["column_prune_agent"].append(prune_eval)
                    if verbose:
                        print(f"  Column Prune (table {table_name}): F1={prune_eval['f1']:.2f}, savings={prune_eval['token_savings_pct']:.1f}%")
                    
                    # Build pruned schemas for SQL generation
                    for table in relevant_tables:
                        try:
                            workspace = self.querygpt.workspaces[workspace_id]
                            full_schema = workspace["table_schemas"][table]
                            prune_res = self.querygpt.column_prune_agent(test_case["question"], table, full_schema)
                            pruned_schemas[table] = prune_res["pruned_schema"]
                        except:
                            # Fallback to full schema
                            workspace = self.querygpt.workspaces[workspace_id]
                            pruned_schemas[table] = workspace["table_schemas"][table]
                
                # Step 4: SQL Generation Agent
                sql_eval = self.evaluate_sql_generation_agent(
                    test_case,
                    workspace_id,
                    relevant_tables,
                    pruned_schemas
                )
                self.results["sql_generation_agent"].append(sql_eval)
                if verbose:
                    print(f"  SQL: similarity={sql_eval['sql_similarity']:.2f}, result_match={sql_eval['result_match']}")
                
                # Overall result
                overall_correct = (
                    intent_eval.get("correct", False) and
                    table_eval.get("correct", False) and
                    sql_eval.get("correct", False)
                )
                self.results["overall"].append({
                    "question": test_case["question"],
                    "correct": overall_correct,
                    "intent_correct": intent_eval.get("correct", False),
                    "table_correct": table_eval.get("correct", False),
                    "sql_correct": sql_eval.get("correct", False)
                })
                
            except Exception as e:
                if verbose:
                    print(f"  ✗ Error: {e}")
                self.results["overall"].append({
                    "question": test_case["question"],
                    "correct": False,
                    "error": str(e)
                })
        
        return self.get_summary()
    
    def get_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics."""
        summary = {}
        
        # Intent Agent
        intent_results = self.results["intent_agent"]
        if intent_results:
            correct = sum(1 for r in intent_results if r.get("correct", False))
            summary["intent_agent"] = {
                "accuracy": correct / len(intent_results),
                "total": len(intent_results),
                "correct": correct,
                "avg_confidence": sum(float(r.get("confidence", 0)) for r in intent_results) / len(intent_results)
            }
        
        # Table Agent
        table_results = self.results["table_agent"]
        if table_results:
            summary["table_agent"] = {
                "avg_precision": sum(r.get("precision", 0) for r in table_results) / len(table_results),
                "avg_recall": sum(r.get("recall", 0) for r in table_results) / len(table_results),
                "avg_f1": sum(r.get("f1", 0) for r in table_results) / len(table_results),
                "total": len(table_results),
                "correct": sum(1 for r in table_results if r.get("correct", False))
            }
        
        # Column Prune Agent
        prune_results = self.results["column_prune_agent"]
        if prune_results:
            summary["column_prune_agent"] = {
                "avg_precision": sum(r.get("precision", 0) for r in prune_results) / len(prune_results),
                "avg_recall": sum(r.get("recall", 0) for r in prune_results) / len(prune_results),
                "avg_f1": sum(r.get("f1", 0) for r in prune_results) / len(prune_results),
                "avg_token_savings": sum(r.get("token_savings_pct", 0) for r in prune_results) / len(prune_results),
                "total": len(prune_results),
                "correct": sum(1 for r in prune_results if r.get("correct", False))
            }
        
        # SQL Generation Agent
        sql_results = self.results["sql_generation_agent"]
        if sql_results:
            summary["sql_generation_agent"] = {
                "avg_sql_similarity": sum(r.get("sql_similarity", 0) for r in sql_results) / len(sql_results),
                "sql_exact_match_rate": sum(1 for r in sql_results if r.get("sql_exact_match", False)) / len(sql_results),
                "result_match_rate": sum(1 for r in sql_results if r.get("result_match", False)) / len(sql_results),
                "total": len(sql_results),
                "correct": sum(1 for r in sql_results if r.get("correct", False))
            }
        
        # Overall
        overall_results = self.results["overall"]
        if overall_results:
            summary["overall"] = {
                "accuracy": sum(1 for r in overall_results if r.get("correct", False)) / len(overall_results),
                "total": len(overall_results),
                "correct": sum(1 for r in overall_results if r.get("correct", False))
            }
        
        # Add latency metrics from agent metrics
        agent_metrics = self.querygpt.get_metrics_summary()
        if agent_metrics:
            summary["latency_metrics"] = {}
            for agent_name, metrics in agent_metrics.items():
                summary["latency_metrics"][agent_name] = {
                    "avg_latency_ms": metrics.get("avg_latency_ms", 0),
                    "min_latency_ms": metrics.get("min_latency_ms", 0),
                    "max_latency_ms": metrics.get("max_latency_ms", 0),
                    "total_tokens": metrics.get("total_tokens", 0),
                    "tool_calls_rate": metrics.get("tool_calls_usage", {}).get("tool_calls_rate", 0)
                }
        
        return summary


class OpenAIQueryGPT(FireworksQueryGPT):
    """OpenAI adapter for QueryGPT - uses OpenAI API instead of Fireworks."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        intent_model: Optional[str] = None,
    ):
        # Initialize OpenAI client (no base_url override)
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        
        # Use OpenAI models
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.intent_model = intent_model or os.getenv("OPENAI_INTENT_MODEL", "gpt-4o-mini")
        self.metrics: List[Any] = []  # Reuse AgentMetrics from parent
        
        # Load workspace data (same as FireworksQueryGPT)
        self.workspaces = self._load_workspaces()
        self.intent_mapping = self._load_intent_mapping()


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate QueryGPT agents against golden dataset")
    parser.add_argument("--provider", choices=["fireworks", "openai"], default="fireworks",
                       help="LLM provider (default: fireworks)")
    parser.add_argument("--model", type=str, help="Model name (overrides env var)")
    parser.add_argument("--intent-model", type=str, help="Intent model name (overrides env var)")
    parser.add_argument("--max-cases", type=int, help="Maximum test cases to evaluate")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    parser.add_argument("--output", type=str, default="evaluation_results_detailed.json",
                       help="Output file for detailed results")
    
    args = parser.parse_args()
    
    # Load golden dataset
    print("Loading golden dataset...")
    with open("evaluation_data.json") as f:
        golden_data = json.load(f)
    
    print(f"Loaded {len(golden_data)} test cases\n")
    
    # Initialize QueryGPT
    print(f"Initializing QueryGPT ({args.provider})...")
    
    if args.provider == "fireworks":
        if not os.getenv("FIREWORKS_API_KEY"):
            print("Error: FIREWORKS_API_KEY not found in .env")
            return
        
        querygpt = FireworksQueryGPT(
            model=args.model,
            intent_model=args.intent_model
        )
        provider_name = "Fireworks"
    else:  # openai
        if not os.getenv("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY not found in .env")
            return
        
        querygpt = OpenAIQueryGPT(
            model=args.model,
            intent_model=args.intent_model
        )
        provider_name = "OpenAI"
    
    print(f"✓ QueryGPT initialized ({provider_name})\n")
    
    # Load database
    print("Loading database...")
    db_conn = load_db()
    print("✓ Database loaded\n")
    
    # Run evaluation
    evaluator = AgentEvaluator(querygpt, db_conn, golden_data)
    summary = evaluator.evaluate_all(max_test_cases=args.max_cases, verbose=not args.quiet)
    
    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}\n")
    print(json.dumps(summary, indent=2))
    
    # Save detailed results
    output_data = {
        "provider": provider_name,
        "summary": summary,
        "detailed_results": evaluator.results,
        "agent_metrics": querygpt.get_metrics_summary()
    }
    
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Detailed results saved to {args.output}")
    
    # Print latency summary
    if "latency_metrics" in summary:
        print(f"\n{'='*80}")
        print("LATENCY METRICS")
        print(f"{'='*80}\n")
        for agent_name, metrics in summary["latency_metrics"].items():
            print(f"{agent_name}:")
            print(f"  Avg Latency: {metrics['avg_latency_ms']:.1f}ms")
            print(f"  Min Latency: {metrics['min_latency_ms']:.1f}ms")
            print(f"  Max Latency: {metrics['max_latency_ms']:.1f}ms")
            print(f"  Tool Calls Rate: {metrics['tool_calls_rate']*100:.1f}%")
            print()
    
    db_conn.close()


if __name__ == "__main__":
    main()

