#!/usr/bin/env python3
"""
Run full evaluation for both Fireworks and OpenAI providers, then compare results.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error running: {description}")
        print(f"Exit code: {result.returncode}")
        return False
    
    print(f"\n✅ Completed: {description}\n")
    return True


def main():
    """Run full evaluation pipeline."""
    
    # Check if we're in the right directory
    if not Path("evaluation_data.json").exists():
        print("Error: evaluation_data.json not found. Are you in the correct directory?")
        sys.exit(1)
    
    # Check if virtual environment is activated
    if not os.getenv("VIRTUAL_ENV"):
        print("⚠️  Warning: Virtual environment not detected.")
        print("Make sure to activate it: source venv_querygpt/bin/activate")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    print("\n" + "="*80)
    print("FULL EVALUATION PIPELINE")
    print("="*80)
    print("\nThis will:")
    print("1. Evaluate Fireworks provider on all test cases")
    print("2. Evaluate OpenAI provider on all test cases")
    print("3. Compare both providers")
    print("\nThis may take several minutes...")
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        sys.exit(0)
    
    # Step 1: Evaluate Fireworks
    fireworks_cmd = [
        "python3", "evaluate_agents.py",
        "--provider", "fireworks",
        "--max-cases", "34",
        "--output", "evaluation_results_fireworks.json"
    ]
    
    if not run_command(fireworks_cmd, "Fireworks Evaluation"):
        print("\n❌ Fireworks evaluation failed. Stopping.")
        sys.exit(1)
    
    # Step 2: Evaluate OpenAI
    openai_cmd = [
        "python3", "evaluate_agents.py",
        "--provider", "openai",
        "--max-cases", "34",
        "--output", "evaluation_results_openai.json"
    ]
    
    if not run_command(openai_cmd, "OpenAI Evaluation"):
        print("\n❌ OpenAI evaluation failed. Stopping.")
        sys.exit(1)
    
    # Step 3: Update compare_providers.py to use the new output files
    # We need to modify compare_providers.py temporarily or create a version that uses these files
    # Actually, let's just update the compare script to check for these files first
    
    # Step 4: Compare providers
    compare_cmd = ["python3", "compare_providers.py"]
    
    # Update compare_providers.py to use the new filenames
    # For now, let's create a wrapper that sets the right filenames
    print("\n" + "="*80)
    print("COMPARING PROVIDERS")
    print("="*80 + "\n")
    
    # Check if files exist
    if not Path("evaluation_results_fireworks.json").exists():
        print("❌ Error: evaluation_results_fireworks.json not found")
        sys.exit(1)
    
    if not Path("evaluation_results_openai.json").exists():
        print("❌ Error: evaluation_results_openai.json not found")
        sys.exit(1)
    
    # Temporarily rename files for compare_providers.py
    # Or update compare_providers.py to accept these filenames
    # Let's update compare_providers.py to check for these files first
    
    # Files are already in the right format for compare_providers.py
    # (it now checks for evaluation_results_fireworks.json first)
    
    if not run_command(compare_cmd, "Provider Comparison"):
        print("\n⚠️  Comparison completed with warnings (check output above)")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print("\nResults saved to:")
    print(f"  - evaluation_results_fireworks.json")
    print(f"  - evaluation_results_openai.json")
    print("\nView comparison above or run: python3 compare_providers.py")


if __name__ == "__main__":
    main()

