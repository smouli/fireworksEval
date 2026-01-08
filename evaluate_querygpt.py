#!/usr/bin/env python3
"""
Evaluate QueryGPT agents' tool calling accuracy and performance.
"""

import json
import re
from typing import Dict, List, Any
from fireworks_querygpt import FireworksQueryGPT
from utils import load_db, query_db


def extract_tables_from_sql(sql: str) -> List[str]:
    """Extract table names from SQL query."""
    # Simple regex - you might want more sophisticated parsing
    tables = set()
    # FROM clause
    tables.update(re.findall(r'FROM\s+(\w+)', sql, re.IGNORECASE))
    # JOIN clauses
    tables.update(re.findall(r'JOIN\s+(\w+)', sql, re.IGNORECASE))
    return list(tables)


def evaluate_tool_calling_accuracy(
    querygpt: FireworksQueryGPT,
    evaluation_data: List[Dict],
    db_conn,
    max_queries: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluate tool calling accuracy for each agent.
    
    Metrics measured:
    - Intent agent: Success rate, confidence scores
    - Table agent: Precision, recall of table selection
    - Column prune agent: Token savings percentage
    - SQL generation: Execution success rate, correctness
    """
    results = {
        "intent_accuracy": [],
        "table_accuracy": [],
        "column_prune_savings": [],
        "sql_execution": [],
        "sql_correctness": [],
        "latency": [],
        "errors": [],
    }
    
    eval_subset = evaluation_data[:max_queries] if max_queries else evaluation_data
    
    for i, test_case in enumerate(eval_subset):
        question = test_case["question"]
        expected_sql = test_case["sql"]
        
        print(f"\n[{i+1}/{len(eval_subset)}] {question}")
        
        try:
            # Run pipeline
            result = querygpt.generate_sql(question)
            
            if "error" in result:
                print(f"  ✗ Error: {result['error']}")
                results["errors"].append({
                    "question": question,
                    "error": result["error"]
                })
                continue
            
            print(f"  ✓ Generated SQL")
            print(f"  Workspace: {result['workspace']}")
            print(f"  Tables: {result['tables']}")
            
            # Evaluate Intent Agent
            # (We don't have expected workspace in test cases, so we'll track confidence)
            if "intent_confidence" in result:
                results["intent_accuracy"].append({
                    "question": question,
                    "workspace": result["workspace"],
                    "confidence": result["intent_confidence"]
                })
            
            # Evaluate Table Agent
            expected_tables = extract_tables_from_sql(expected_sql)
            selected_tables = set(result["tables"])
            expected_tables_set = set(expected_tables)
            
            if expected_tables_set:
                table_precision = len(selected_tables & expected_tables_set) / len(selected_tables) if selected_tables else 0
                table_recall = len(selected_tables & expected_tables_set) / len(expected_tables_set) if expected_tables_set else 0
                
                results["table_accuracy"].append({
                    "question": question,
                    "expected": list(expected_tables_set),
                    "selected": list(selected_tables),
                    "precision": table_precision,
                    "recall": table_recall,
                    "f1": 2 * (table_precision * table_recall) / (table_precision + table_recall) if (table_precision + table_recall) > 0 else 0
                })
                
                print(f"  Table Precision: {table_precision:.2f}, Recall: {table_recall:.2f}")
            
            # Evaluate Column Prune Agent
            if "metrics" in result and result["metrics"].get("prune"):
                prune_metric = result["metrics"]["prune"]
                if hasattr(prune_metric, "accuracy") and prune_metric.accuracy:
                    results["column_prune_savings"].append({
                        "question": question,
                        "token_savings_pct": prune_metric.accuracy
                    })
            
            # Evaluate SQL Execution
            try:
                actual_results = query_db(db_conn, result["sql"], return_as_df=False)
                results["sql_execution"].append({
                    "question": question,
                    "executed": True,
                    "rows_returned": len(actual_results),
                    "sql": result["sql"]
                })
                print(f"  ✓ SQL executed successfully ({len(actual_results)} rows)")
            except Exception as e:
                results["sql_execution"].append({
                    "question": question,
                    "executed": False,
                    "error": str(e),
                    "sql": result["sql"]
                })
                print(f"  ✗ SQL execution failed: {e}")
            
            # Evaluate SQL Correctness (compare results)
            try:
                expected_results = test_case.get("expected_result", [])
                if expected_results:
                    actual_results = query_db(db_conn, result["sql"], return_as_df=False)
                    
                    # Simple comparison - check row count and basic structure
                    # For more sophisticated comparison, you'd want to compare actual values
                    rows_match = len(actual_results) == len(expected_results)
                    
                    # Check if column names match (if any results)
                    columns_match = False
                    if actual_results and expected_results:
                        actual_cols = set(actual_results[0].keys())
                        expected_cols = set(expected_results[0].keys())
                        columns_match = actual_cols == expected_cols
                    
                    results["sql_correctness"].append({
                        "question": question,
                        "rows_match": rows_match,
                        "columns_match": columns_match,
                        "expected_rows": len(expected_results),
                        "actual_rows": len(actual_results),
                        "expected_sql": expected_sql,
                        "generated_sql": result["sql"]
                    })
                    
                    if rows_match and columns_match:
                        print(f"  ✓ Results match expected ({len(actual_results)} rows)")
                    else:
                        print(f"  ⚠ Results differ: expected {len(expected_results)} rows, got {len(actual_results)}")
            except Exception as e:
                print(f"  ⚠ Could not compare results: {e}")
            
            results["latency"].append(result["total_latency_ms"])
            print(f"  Total latency: {result['total_latency_ms']:.0f}ms")
            
        except Exception as e:
            print(f"  ✗ Pipeline error: {e}")
            results["errors"].append({
                "question": question,
                "error": str(e)
            })
    
    # Calculate summary metrics
    summary = {
        "tool_calling_success_rate": {},
        "table_selection": {},
        "column_prune": {},
        "sql_execution_rate": 0,
        "sql_correctness_rate": 0,
        "avg_latency_ms": 0,
        "agent_metrics": querygpt.get_metrics_summary(),
    }
    
    # Tool calling success rates from agent metrics
    agent_metrics = querygpt.get_metrics_summary()
    for agent_name in ["intent_agent", "table_agent", "column_prune_agent", "sql_generation_agent"]:
        if agent_name in agent_metrics:
            summary["tool_calling_success_rate"][agent_name] = agent_metrics[agent_name]["success_rate"]
    
    # Table selection metrics
    if results["table_accuracy"]:
        summary["table_selection"] = {
            "avg_precision": sum(r["precision"] for r in results["table_accuracy"]) / len(results["table_accuracy"]),
            "avg_recall": sum(r["recall"] for r in results["table_accuracy"]) / len(results["table_accuracy"]),
            "avg_f1": sum(r["f1"] for r in results["table_accuracy"]) / len(results["table_accuracy"]),
            "total_evaluated": len(results["table_accuracy"])
        }
    
    # Column prune savings
    if results["column_prune_savings"]:
        summary["column_prune"] = {
            "avg_token_savings_pct": sum(r["token_savings_pct"] for r in results["column_prune_savings"]) / len(results["column_prune_savings"]),
            "total_evaluated": len(results["column_prune_savings"])
        }
    
    # SQL execution rate
    if results["sql_execution"]:
        summary["sql_execution_rate"] = sum(1 for r in results["sql_execution"] if r["executed"]) / len(results["sql_execution"])
    
    # SQL correctness rate
    if results["sql_correctness"]:
        correct = sum(1 for r in results["sql_correctness"] if r["rows_match"] and r["columns_match"])
        summary["sql_correctness_rate"] = correct / len(results["sql_correctness"])
    
    # Average latency
    if results["latency"]:
        summary["avg_latency_ms"] = sum(results["latency"]) / len(results["latency"])
    
    # Error rate
    summary["error_rate"] = len(results["errors"]) / len(eval_subset) if eval_subset else 0
    
    return {
        "summary": summary,
        "detailed_results": results
    }


if __name__ == "__main__":
    import sys
    
    # Load evaluation data
    with open("evaluation_data.json") as f:
        eval_data = json.load(f)
    
    # Initialize QueryGPT
    print("Initializing Fireworks QueryGPT...")
    querygpt = FireworksQueryGPT()
    
    # Load database
    print("Loading database...")
    db_conn = load_db()
    
    # Determine how many queries to evaluate
    max_queries = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    
    # Run evaluation
    print(f"\nEvaluating QueryGPT tool calling accuracy on {max_queries} queries...")
    print("=" * 80)
    results = evaluate_tool_calling_accuracy(querygpt, eval_data, db_conn, max_queries=max_queries)
    
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(json.dumps(results["summary"], indent=2))
    
    # Save detailed results
    output_file = "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {output_file}")
    
    db_conn.close()

