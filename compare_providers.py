#!/usr/bin/env python3
"""
Compare Fireworks vs OpenAI providers on latency and accuracy.
"""

import json
from pathlib import Path


def load_results(filepath: str):
    """Load evaluation results from JSON file."""
    with open(filepath) as f:
        return json.load(f)


def compare_providers():
    """Compare Fireworks and OpenAI evaluation results."""
    
    # Try multiple possible filenames
    fireworks_candidates = [
        "evaluation_results_fireworks.json",
        "evaluation_results_detailed.json"
    ]
    openai_candidates = [
        "evaluation_results_openai.json"
    ]
    
    fireworks_file = None
    for candidate in fireworks_candidates:
        if Path(candidate).exists():
            fireworks_file = candidate
            break
    
    openai_file = None
    for candidate in openai_candidates:
        if Path(candidate).exists():
            openai_file = candidate
            break
    
    if not fireworks_file:
        print(f"Error: Fireworks results not found. Tried: {fireworks_candidates}")
        print("Run evaluation with --provider fireworks first.")
        return
    
    if not openai_file:
        print(f"Error: OpenAI results not found. Tried: {openai_candidates}")
        print("Run evaluation with --provider openai first.")
        return
    
    print(f"Using Fireworks results: {fireworks_file}")
    print(f"Using OpenAI results: {openai_file}\n")
    
    fireworks_results = load_results(fireworks_file)
    openai_results = load_results(openai_file)
    
    print("\n" + "="*80)
    print("PROVIDER COMPARISON: Fireworks vs OpenAI")
    print("="*80 + "\n")
    
    # Overall Accuracy
    print("📊 OVERALL ACCURACY")
    print("-"*80)
    fw_overall = fireworks_results["summary"]["overall"]["accuracy"]
    oa_overall = openai_results["summary"]["overall"]["accuracy"]
    diff = oa_overall - fw_overall
    print(f"Fireworks: {fw_overall*100:.1f}% ({fireworks_results['summary']['overall']['correct']}/{fireworks_results['summary']['overall']['total']})")
    print(f"OpenAI:    {oa_overall*100:.1f}% ({openai_results['summary']['overall']['correct']}/{openai_results['summary']['overall']['total']})")
    print(f"Difference: {diff*100:+.1f}% ({'OpenAI' if diff > 0 else 'Fireworks'} wins)")
    print()
    
    # Agent Accuracy Comparison
    print("🤖 AGENT ACCURACY COMPARISON")
    print("-"*80)
    
    agents = ["intent_agent", "table_agent", "sql_generation_agent"]
    for agent in agents:
        if agent in fireworks_results["summary"] and agent in openai_results["summary"]:
            fw_data = fireworks_results["summary"][agent]
            oa_data = openai_results["summary"][agent]
            
            if "accuracy" in fw_data:
                fw_acc = fw_data["accuracy"]
                oa_acc = oa_data["accuracy"]
                diff = oa_acc - fw_acc
                print(f"\n{agent}:")
                print(f"  Fireworks: {fw_acc*100:.1f}%")
                print(f"  OpenAI:    {oa_acc*100:.1f}%")
                print(f"  Difference: {diff*100:+.1f}%")
            elif "avg_f1" in fw_data:
                fw_f1 = fw_data["avg_f1"]
                oa_f1 = oa_data["avg_f1"]
                diff = oa_f1 - fw_f1
                print(f"\n{agent} (F1 Score):")
                print(f"  Fireworks: {fw_f1*100:.1f}%")
                print(f"  OpenAI:    {oa_f1*100:.1f}%")
                print(f"  Difference: {diff*100:+.1f}%")
    
    # Latency Comparison
    print("\n" + "="*80)
    print("⚡ LATENCY COMPARISON (Lower is Better)")
    print("="*80 + "\n")
    
    if "latency_metrics" in fireworks_results["summary"] and "latency_metrics" in openai_results["summary"]:
        fw_latency = fireworks_results["summary"]["latency_metrics"]
        oa_latency = openai_results["summary"]["latency_metrics"]
        
        for agent_name in fw_latency.keys():
            if agent_name in oa_latency:
                fw_avg = fw_latency[agent_name]["avg_latency_ms"]
                oa_avg = oa_latency[agent_name]["avg_latency_ms"]
                speedup = fw_avg / oa_avg if oa_avg > 0 else 0
                
                print(f"{agent_name}:")
                print(f"  Fireworks: {fw_avg:.1f}ms (min: {fw_latency[agent_name]['min_latency_ms']:.1f}ms, max: {fw_latency[agent_name]['max_latency_ms']:.1f}ms)")
                print(f"  OpenAI:    {oa_avg:.1f}ms (min: {oa_latency[agent_name]['min_latency_ms']:.1f}ms, max: {oa_latency[agent_name]['max_latency_ms']:.1f}ms)")
                
                if speedup < 1:
                    print(f"  🚀 Fireworks is {1/speedup:.2f}x faster")
                elif speedup > 1:
                    print(f"  🚀 OpenAI is {speedup:.2f}x faster")
                else:
                    print(f"  ⚖️  Similar performance")
                print()
        
        # Total pipeline latency
        fw_total = sum(m["avg_latency_ms"] for m in fw_latency.values())
        oa_total = sum(m["avg_latency_ms"] for m in oa_latency.values())
        total_speedup = fw_total / oa_total if oa_total > 0 else 0
        
        print("Total Pipeline Latency:")
        print(f"  Fireworks: {fw_total:.1f}ms")
        print(f"  OpenAI:    {oa_total:.1f}ms")
        if total_speedup < 1:
            print(f"  🚀 Fireworks is {1/total_speedup:.2f}x faster overall")
        elif total_speedup > 1:
            print(f"  🚀 OpenAI is {total_speedup:.2f}x faster overall")
        print()
    
    # Tool Calls Usage
    print("="*80)
    print("🔧 TOOL CALLS USAGE & ACCURACY")
    print("="*80 + "\n")
    
    if "agent_metrics" in fireworks_results and "agent_metrics" in openai_results:
        fw_metrics = fireworks_results["agent_metrics"]
        oa_metrics = openai_results["agent_metrics"]
        
        for agent_name in fw_metrics.keys():
            if agent_name in oa_metrics:
                fw_tc = fw_metrics[agent_name].get("tool_calls_usage", {})
                oa_tc = oa_metrics[agent_name].get("tool_calls_usage", {})
                fw_acc = fw_metrics[agent_name].get("tool_call_accuracy", {})
                oa_acc = oa_metrics[agent_name].get("tool_call_accuracy", {})
                
                fw_rate = fw_tc.get("tool_calls_rate", 0)
                oa_rate = oa_tc.get("tool_calls_rate", 0)
                fw_acc_rate = fw_acc.get("accuracy_rate", 0)
                oa_acc_rate = oa_acc.get("accuracy_rate", 0)
                
                print(f"{agent_name}:")
                print(f"  Structured Tool Calls Rate:")
                print(f"    Fireworks: {fw_rate*100:.1f}% ({fw_tc.get('structured_tool_calls', 0)}/{fw_tc.get('structured_tool_calls', 0) + fw_tc.get('fallback_parsing', 0)})")
                print(f"    OpenAI:    {oa_rate*100:.1f}% ({oa_tc.get('structured_tool_calls', 0)}/{oa_tc.get('structured_tool_calls', 0) + oa_tc.get('fallback_parsing', 0)})")
                print(f"  Tool Call Accuracy (schema validation):")
                print(f"    Fireworks: {fw_acc_rate*100:.1f}% ({fw_acc.get('accurate_calls', 0)}/{fw_tc.get('structured_tool_calls', 0) + fw_tc.get('fallback_parsing', 0)})")
                print(f"    OpenAI:    {oa_acc_rate*100:.1f}% ({oa_acc.get('accurate_calls', 0)}/{oa_tc.get('structured_tool_calls', 0) + oa_tc.get('fallback_parsing', 0)})")
                print(f"  Validation Errors:")
                print(f"    Fireworks: {fw_acc.get('total_validation_errors', 0)} total ({fw_acc.get('avg_validation_errors_per_call', 0):.2f} avg per call)")
                print(f"    OpenAI:    {oa_acc.get('total_validation_errors', 0)} total ({oa_acc.get('avg_validation_errors_per_call', 0):.2f} avg per call)")
                print()
    
    # SQL Generation Quality
    print("="*80)
    print("📝 SQL GENERATION QUALITY")
    print("="*80 + "\n")
    
    fw_sql = fireworks_results["summary"]["sql_generation_agent"]
    oa_sql = openai_results["summary"]["sql_generation_agent"]
    
    print(f"SQL Similarity:")
    print(f"  Fireworks: {fw_sql['avg_sql_similarity']*100:.1f}%")
    print(f"  OpenAI:    {oa_sql['avg_sql_similarity']*100:.1f}%")
    print()
    
    print(f"Exact SQL Match Rate:")
    print(f"  Fireworks: {fw_sql['sql_exact_match_rate']*100:.1f}%")
    print(f"  OpenAI:    {oa_sql['sql_exact_match_rate']*100:.1f}%")
    print()
    
    print(f"Result Match Rate:")
    print(f"  Fireworks: {fw_sql['result_match_rate']*100:.1f}%")
    print(f"  OpenAI:    {oa_sql['result_match_rate']*100:.1f}%")
    print()
    
    print("="*80)


if __name__ == "__main__":
    compare_providers()

