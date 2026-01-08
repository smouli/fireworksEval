#!/usr/bin/env python3
"""
Evaluate fine-tuned Column Prune Agent.
Compares baseline (tool calling) vs fine-tuned (JSON mode).
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from evaluate_agents import AgentEvaluator, FireworksQueryGPT, load_env_file
from utils import load_db

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

load_env_file()

def evaluate_column_prune_agent(
    fine_tuned_model: str,
    evaluation_data_path: str = "evaluation_data.json",
    output_path: str = "column_prune_evaluation.json"
):
    """Evaluate fine-tuned column prune agent."""
    
    print("="*80)
    print("EVALUATING FINE-TUNED COLUMN PRUNE AGENT")
    print("="*80)
    print(f"Fine-tuned Model: {fine_tuned_model}")
    print(f"Evaluation Data: {evaluation_data_path}")
    
    # Load evaluation data
    with open(evaluation_data_path, 'r') as f:
        eval_data = json.load(f)
    
    # Load database
    db_path = os.getenv("DATABASE_PATH", "uber.db")
    db_conn = load_db(db_path)
    
    # Initialize QueryGPT with fine-tuned model for column pruning
    print(f"\n🔧 Initializing QueryGPT with fine-tuned column prune model...")
    querygpt_finetuned = FireworksQueryGPT(
        column_prune_model=fine_tuned_model  # This enables JSON mode for column pruning
    )
    
    # Also create baseline (tool calling) for comparison
    print(f"🔧 Initializing QueryGPT baseline (tool calling)...")
    querygpt_baseline = FireworksQueryGPT()
    
    # Evaluate both
    print(f"\n📊 Evaluating baseline (tool calling)...")
    evaluator_baseline = AgentEvaluator(querygpt_baseline, db_conn, eval_data)
    results_baseline = evaluator_baseline.evaluate_all()
    
    print(f"\n📊 Evaluating fine-tuned (JSON mode)...")
    evaluator_finetuned = AgentEvaluator(querygpt_finetuned, db_conn, eval_data)
    results_finetuned = evaluator_finetuned.evaluate_all()
    
    # Compare results
    print(f"\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    baseline_col_prune = results_baseline["summary"].get("column_prune_agent", {})
    finetuned_col_prune = results_finetuned["summary"].get("column_prune_agent", {})
    
    baseline_f1 = baseline_col_prune.get("avg_f1", 0)
    finetuned_f1 = finetuned_col_prune.get("avg_f1", 0)
    
    baseline_precision = baseline_col_prune.get("avg_precision", 0)
    finetuned_precision = finetuned_col_prune.get("avg_precision", 0)
    
    baseline_recall = baseline_col_prune.get("avg_recall", 0)
    finetuned_recall = finetuned_col_prune.get("avg_recall", 0)
    
    baseline_correct = baseline_col_prune.get("correct", 0)
    finetuned_correct = finetuned_col_prune.get("correct", 0)
    
    baseline_total = baseline_col_prune.get("total", 0)
    finetuned_total = finetuned_col_prune.get("total", 0)
    
    print(f"\n📋 Column Prune Agent Metrics:")
    print(f"   Metric              | Baseline    | Fine-tuned  | Change")
    print(f"   --------------------|-------------|-------------|----------")
    print(f"   F1 Score            | {baseline_f1*100:6.2f}%    | {finetuned_f1*100:6.2f}%    | {finetuned_f1-baseline_f1:+.2f}%")
    print(f"   Precision           | {baseline_precision*100:6.2f}%    | {finetuned_precision*100:6.2f}%    | {finetuned_precision-baseline_precision:+.2f}%")
    print(f"   Recall              | {baseline_recall*100:6.2f}%    | {finetuned_recall*100:6.2f}%    | {finetuned_recall-baseline_recall:+.2f}%")
    print(f"   Correct             | {baseline_correct:3d}/{baseline_total:3d}      | {finetuned_correct:3d}/{finetuned_total:3d}      | {finetuned_correct-baseline_correct:+d}")
    
    # Latency comparison
    baseline_latency = results_baseline["summary"].get("latency_metrics", {}).get("column_prune_agent", {})
    finetuned_latency = results_finetuned["summary"].get("latency_metrics", {}).get("column_prune_agent", {})
    
    baseline_avg_latency = baseline_latency.get("avg_latency_ms", 0)
    finetuned_avg_latency = finetuned_latency.get("avg_latency_ms", 0)
    
    baseline_tokens = baseline_latency.get("total_tokens", 0)
    finetuned_tokens = finetuned_latency.get("total_tokens", 0)
    
    print(f"\n⚡ Performance Metrics:")
    print(f"   Metric              | Baseline    | Fine-tuned  | Change")
    print(f"   --------------------|-------------|-------------|----------")
    print(f"   Avg Latency (ms)    | {baseline_avg_latency:8.1f}  | {finetuned_avg_latency:8.1f}  | {finetuned_avg_latency-baseline_avg_latency:+.1f} ms")
    print(f"   Total Tokens         | {baseline_tokens:8d}  | {finetuned_tokens:8d}  | {finetuned_tokens-baseline_tokens:+d}")
    
    # Overall accuracy comparison
    baseline_overall = results_baseline["summary"].get("overall", {}).get("accuracy", 0)
    finetuned_overall = results_finetuned["summary"].get("overall", {}).get("accuracy", 0)
    
    print(f"\n🎯 Overall Pipeline Accuracy:")
    print(f"   Baseline:   {baseline_overall*100:.2f}%")
    print(f"   Fine-tuned: {finetuned_overall*100:.2f}%")
    print(f"   Change:     {finetuned_overall-baseline_overall:+.2f}%")
    
    # Save results
    comparison_results = {
        "baseline": results_baseline,
        "finetuned": results_finetuned,
        "comparison": {
            "column_prune_agent": {
                "f1_score": {
                    "baseline": baseline_f1,
                    "finetuned": finetuned_f1,
                    "improvement": finetuned_f1 - baseline_f1,
                    "improvement_pct": ((finetuned_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0
                },
                "precision": {
                    "baseline": baseline_precision,
                    "finetuned": finetuned_precision,
                    "improvement": finetuned_precision - baseline_precision
                },
                "recall": {
                    "baseline": baseline_recall,
                    "finetuned": finetuned_recall,
                    "improvement": finetuned_recall - baseline_recall
                },
                "correct": {
                    "baseline": baseline_correct,
                    "finetuned": finetuned_correct,
                    "improvement": finetuned_correct - baseline_correct
                }
            },
            "latency": {
                "baseline_avg_ms": baseline_avg_latency,
                "finetuned_avg_ms": finetuned_avg_latency,
                "improvement_ms": finetuned_avg_latency - baseline_avg_latency
            },
            "tokens": {
                "baseline": baseline_tokens,
                "finetuned": finetuned_tokens,
                "improvement": finetuned_tokens - baseline_tokens
            },
            "overall_accuracy": {
                "baseline": baseline_overall,
                "finetuned": finetuned_overall,
                "improvement": finetuned_overall - baseline_overall
            }
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_path}")
    
    # Final verdict
    print(f"\n" + "="*80)
    print("VERDICT")
    print("="*80)
    if finetuned_f1 > baseline_f1:
        print(f"✅ Fine-tuning IMPROVED F1 score by {((finetuned_f1 - baseline_f1) / baseline_f1 * 100):.1f}%")
    elif finetuned_f1 < baseline_f1:
        print(f"❌ Fine-tuning DECREASED F1 score by {((baseline_f1 - finetuned_f1) / baseline_f1 * 100):.1f}%")
    else:
        print(f"➡️  No significant change in F1 score")
    
    return comparison_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Column Prune Agent")
    parser.add_argument("--model", required=True, help="Fine-tuned model ID (e.g., accounts/.../models/column-prune-agent-json)")
    parser.add_argument("--eval-data", default="evaluation_data.json", help="Path to evaluation data")
    parser.add_argument("--output", default="column_prune_evaluation.json", help="Output file for results")
    
    args = parser.parse_args()
    
    evaluate_column_prune_agent(
        fine_tuned_model=args.model,
        evaluation_data_path=args.eval_data,
        output_path=args.output
    )

