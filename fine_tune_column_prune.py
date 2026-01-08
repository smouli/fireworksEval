#!/usr/bin/env python3
"""
Fine-tune Column Prune Agent in JSON mode.
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from typing import Dict, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from run_sft_workflow import FireworksAPIClient

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_MODEL = "accounts/fireworks/models/qwen3-30b-a3b-instruct-2507"

def fine_tune_column_prune_agent(
    dataset_path: str = "column_prune_dataset.jsonl",
    base_model: str = BASE_MODEL,
    output_model_id: str = "column-prune-agent-json"
):
    """Fine-tune column prune agent on JSON mode dataset."""
    
    api_key = os.getenv("FIREWORKS_API_KEY")
    account_id = os.getenv("FIREWORKS_ACCOUNT_ID")
    
    if not api_key:
        raise ValueError("FIREWORKS_API_KEY must be set in environment")
    if not account_id:
        raise ValueError("FIREWORKS_ACCOUNT_ID must be set in environment")
    
    client = FireworksAPIClient(api_key, account_id)
    
    print("="*80)
    print("FINE-TUNING COLUMN PRUNE AGENT (JSON MODE)")
    print("="*80)
    print(f"Base Model: {base_model}")
    print(f"Output Model ID: {output_model_id}")
    print(f"Dataset: {dataset_path}")
    
    # Check dataset file exists
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # Count examples
    with open(dataset_path, 'r') as f:
        example_count = sum(1 for line in f)
    print(f"Dataset contains {example_count} examples")
    
    # 1. Create and upload dataset
    print(f"\n📦 Creating dataset...")
    dataset_id = f"{output_model_id}-dataset"
    
    # Check if dataset exists and is ready
    try:
        dataset = client.get_dataset(dataset_id)
        if dataset.get("state") == "READY":
            existing_count = dataset.get("exampleCount", 0)
            if existing_count == example_count:
                print(f"✅ Dataset {dataset_id} already exists and is READY with {existing_count} examples")
                dataset_name = dataset.get("name") or dataset.get("datasetId") or dataset_id
            else:
                # Use existing dataset anyway (might have been updated)
                print(f"⚠️  Dataset {dataset_id} exists with {existing_count} examples (expected {example_count}), using it anyway")
                dataset_name = dataset.get("name") or dataset.get("datasetId") or dataset_id
        else:
            print(f"⚠️  Dataset {dataset_id} exists but state is {dataset.get('state')}, waiting for READY...")
            # Wait for it to become READY
            import time
            max_wait = 300
            start_time = time.time()
            while time.time() - start_time < max_wait:
                dataset = client.get_dataset(dataset_id)
                if dataset.get("state") == "READY":
                    break
                time.sleep(5)
            dataset_name = dataset.get("name") or dataset.get("datasetId") or dataset_id
    except Exception as e:
        # Dataset doesn't exist, create it
        from run_sft_workflow import create_and_upload_dataset
        print(f"📦 Creating and uploading dataset...")
        dataset = create_and_upload_dataset(client, dataset_id, dataset_path)
        dataset_name = dataset.get("name") or dataset.get("datasetId") or dataset_id
        print(f"✅ Dataset ready: {dataset_name}")
    
    # 2. Create fine-tuning job
    print(f"\n🚀 Creating fine-tuning job...")
    fine_tuned_model_id = f"accounts/{account_id}/models/{output_model_id}"
    
    # Get dataset name from response
    dataset_name = dataset.get("name") or dataset.get("datasetId") or dataset_id
    
    job = client.create_sft_job(
        dataset_name=dataset_name,
        base_model=base_model,
        fine_tuned_model_id=fine_tuned_model_id,
        config={
            "learning_rate": 5e-6,
            "learning_rate_warmup_steps": 100,
            "epochs": 2,
            "lora_rank": 32,
            "gradient_accumulation_steps": 2,
            "batch_size": 16384,
            "max_context_length": 16384,
        }
    )
    
    job_id = job.get("id") or job.get("name")
    if not job_id:
        print(f"⚠️  Warning: Job response doesn't contain 'id' or 'name' field")
        print(f"   Response: {json.dumps(job, indent=2)}")
        print(f"\n   Please check the Fireworks dashboard for job status:")
        print(f"   https://app.fireworks.ai/fine-tuning")
        print(f"\n   Once the job completes, you can evaluate with:")
        print(f"   python3 evaluate_column_prune_finetuned.py --model {fine_tuned_model_id}")
        return fine_tuned_model_id
    
    print(f"✅ Created fine-tuning job: {job_id}")
    print(f"   Job ID: {job_id}")
    print(f"   Output Model: {fine_tuned_model_id}")
    
    # 3. Monitor job
    print(f"\n⏳ Monitoring fine-tuning job...")
    print(f"   (Check status at: https://app.fireworks.ai/fine-tuning)")
    while True:
        job_status = client.get_sft_job(job_id)
        state = job_status.get("state")
        print(f"   State: {state}")
        
        if state == "COMPLETED":
            print(f"\n✅ Fine-tuning completed!")
            print(f"   Model: {fine_tuned_model_id}")
            break
        elif state == "FAILED":
            error = job_status.get("error", "Unknown error")
            raise RuntimeError(f"Fine-tuning failed: {error}")
        
        time.sleep(30)
    
    return fine_tuned_model_id

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune Column Prune Agent")
    parser.add_argument("--dataset", default="column_prune_dataset.jsonl", help="Path to dataset JSONL file")
    parser.add_argument("--base-model", default=BASE_MODEL, help="Base model to fine-tune")
    parser.add_argument("--output-id", default="column-prune-agent-json", help="Output model ID")
    
    args = parser.parse_args()
    
    fine_tuned_model = fine_tune_column_prune_agent(
        dataset_path=args.dataset,
        base_model=args.base_model,
        output_model_id=args.output_id
    )
    print(f"\n✅ Fine-tuned model ready: {fine_tuned_model}")
    print(f"\nNext steps:")
    print(f"1. Deploy the model in Fireworks dashboard if needed")
    print(f"2. Run evaluation with: python3 evaluate_column_prune_finetuned.py --model {fine_tuned_model}")

