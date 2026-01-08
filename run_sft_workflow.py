#!/usr/bin/env python3
"""
Complete SFT Workflow for Fireworks AI:
1. Baseline evaluation (before tuning)
2. Upload dataset for fine-tuning
3. Create and monitor fine-tuning job
4. Deploy fine-tuned model
5. Post-tuning evaluation
6. Compare results (accuracy, latency, cost)
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Check and import required dependencies
try:
    import requests
except ImportError:
    print("Error: 'requests' module not found.")
    print("\nPlease install it using one of the following:")
    print("  1. Activate your virtual environment and run:")
    print("     source venv_querygpt/bin/activate")
    print("     pip install requests")
    print("\n  2. Or install globally:")
    print("     pip3 install requests")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("Warning: 'python-dotenv' not found. .env file loading may not work.")
    print("Install with: pip install python-dotenv")
    load_dotenv = lambda *args, **kwargs: None

# Use REST API directly (more reliable than SDK)
import requests
from typing import Optional as TypingOptional

from evaluate_agents import AgentEvaluator, FireworksQueryGPT, load_env_file
from utils import load_db

# Load environment variables
load_dotenv()
load_env_file()

# Constants
# Note: Dataset IDs must follow resource ID restrictions:
# - Only lowercase letters, numbers, and hyphens
# - Must start with a letter
# - Max 63 characters
DATASET_ID = "querygpt-sft-dataset-v2"  # Use the 1000-example dataset
FINE_TUNED_MODEL_PREFIX = "querygpt-sft"
# Default to a tunable model (Qwen models are typically tunable)
# Note: Some models like kimi-k2-thinking may not support fine-tuning
# Use qwen3-30b as default (known to work with fine-tuning)
# Note: qwen2.5-7b-instruct format has dots which API doesn't accept
BASE_MODEL = os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/qwen3-30b-a3b-instruct-2507")
EVALUATION_DATA_FILE = "evaluation_data.json"
GOLDEN_DATASET_FILE = "golden_dataset_sft.jsonl"
SFT_CONFIG = {
    "learning_rate": 5e-6,  # Lower learning rate for more stable training
    "learning_rate_warmup_steps": 100,  # Reduced warmup steps
    "epochs": 2,  # More epochs for better convergence
    "lora_rank": 32,  # Higher rank for better capacity
    "gradient_accumulation_steps": 2,
    "batch_size": 16384,  # Must be >= max_context_length (16384)
    "max_context_length": 16384,
}


def calculate_reward_metrics(evaluation_results: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate reward function metrics from evaluation results.
    Returns a dictionary with accuracy, latency, and cost metrics.
    """
    summary = evaluation_results.get("summary", {})
    
    # Accuracy metrics (weighted average)
    accuracy_weights = {
        "intent_agent": 0.15,
        "table_agent": 0.20,
        "column_prune_agent": 0.15,
        "sql_generation_agent": 0.30,
        "overall": 0.20,
    }
    
    accuracy_score = 0.0
    for agent, weight in accuracy_weights.items():
        if agent in summary:
            if agent == "overall":
                acc = summary[agent].get("accuracy", 0.0)
            elif "avg_f1" in summary[agent]:
                acc = summary[agent]["avg_f1"]
            elif "accuracy" in summary[agent]:
                acc = summary[agent]["accuracy"]
            elif "avg_sql_similarity" in summary[agent]:
                acc = summary[agent]["avg_sql_similarity"]
            else:
                acc = 0.0
            accuracy_score += acc * weight
    
    # Latency metrics (lower is better, normalized)
    latency_metrics = summary.get("latency_metrics", {})
    total_latency_ms = 0.0
    latency_count = 0
    
    for agent_name, metrics in latency_metrics.items():
        avg_latency = metrics.get("avg_latency_ms", 0)
        total_latency_ms += avg_latency
        latency_count += 1
    
    avg_latency_ms = total_latency_ms / latency_count if latency_count > 0 else 0.0
    
    # Cost estimation (based on token usage)
    agent_metrics = evaluation_results.get("agent_metrics", {})
    total_tokens = 0
    for agent_name, metrics in agent_metrics.items():
        total_tokens += metrics.get("total_tokens", 0)
    
    # Rough cost estimation (assuming $0.50 per 1M tokens for inference)
    # This is a placeholder - actual cost depends on Fireworks pricing
    cost_per_million_tokens = 0.50
    estimated_cost = (total_tokens / 1_000_000) * cost_per_million_tokens
    
    # Composite reward score (higher is better)
    # Normalize latency: assume 1000ms is baseline, lower is better
    latency_score = max(0, 1.0 - (avg_latency_ms / 1000.0))
    
    # Normalize cost: assume $1 is baseline, lower is better
    cost_score = max(0, 1.0 - min(estimated_cost, 1.0))
    
    # Weighted composite score
    reward_score = (
        accuracy_score * 0.60 +  # 60% weight on accuracy
        latency_score * 0.25 +   # 25% weight on latency
        cost_score * 0.15         # 15% weight on cost
    )
    
    return {
        "accuracy_score": accuracy_score,
        "avg_latency_ms": avg_latency_ms,
        "latency_score": latency_score,
        "total_tokens": total_tokens,
        "estimated_cost": estimated_cost,
        "cost_score": cost_score,
        "reward_score": reward_score,
    }


def run_baseline_evaluation() -> Dict[str, Any]:
    """Run baseline evaluation before fine-tuning."""
    print("\n" + "="*80)
    print("STEP 1: BASELINE EVALUATION (Before Fine-Tuning)")
    print("="*80)
    
    if not os.getenv("FIREWORKS_API_KEY"):
        raise ValueError("FIREWORKS_API_KEY not found in environment")
    
    # Load evaluation data
    print(f"\nLoading evaluation data from {EVALUATION_DATA_FILE}...")
    with open(EVALUATION_DATA_FILE) as f:
        golden_data = json.load(f)
    print(f"✓ Loaded {len(golden_data)} test cases")
    
    # Initialize QueryGPT with base model
    print(f"\nInitializing QueryGPT with base model: {BASE_MODEL}")
    querygpt = FireworksQueryGPT(model=BASE_MODEL)
    print("✓ QueryGPT initialized")
    
    # Load database
    print("\nLoading database...")
    db_conn = load_db()
    print("✓ Database loaded")
    
    # Run evaluation
    print("\nRunning evaluation...")
    evaluator = AgentEvaluator(querygpt, db_conn, golden_data)
    summary = evaluator.evaluate_all(verbose=False)
    
    # Calculate reward metrics
    evaluation_results = {
        "summary": summary,
        "detailed_results": evaluator.results,
        "agent_metrics": querygpt.get_metrics_summary(),
    }
    
    reward_metrics = calculate_reward_metrics(evaluation_results)
    evaluation_results["reward_metrics"] = reward_metrics
    
    # Save baseline results
    baseline_file = "evaluation_results_baseline.json"
    with open(baseline_file, "w") as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"\n✓ Baseline evaluation saved to {baseline_file}")
    
    # Print summary
    print("\n" + "-"*80)
    print("BASELINE METRICS:")
    print("-"*80)
    print(f"Overall Accuracy: {summary['overall']['accuracy']*100:.1f}%")
    print(f"Average Latency: {reward_metrics['avg_latency_ms']:.1f}ms")
    print(f"Total Tokens: {reward_metrics['total_tokens']:,}")
    print(f"Estimated Cost: ${reward_metrics['estimated_cost']:.4f}")
    print(f"Reward Score: {reward_metrics['reward_score']:.3f}")
    
    db_conn.close()
    return evaluation_results


class FireworksAPIClient:
    """Simple REST API client for Fireworks AI."""
    
    def __init__(self, api_key: str, account_id: Optional[str] = None):
        self.api_key = api_key
        self.account_id = account_id or os.getenv("FIREWORKS_ACCOUNT_ID", "")
        # Base URL for Fireworks AI API (without /inference)
        self.base_url = "https://api.fireworks.ai"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        # Verify account ID if provided
        if self.account_id:
            print(f"Using account ID: {self.account_id}")
        else:
            print("⚠ Warning: FIREWORKS_ACCOUNT_ID not set. Attempting to infer from API...")
            self.account_id = self._get_account_id()
    
    def _get_account_id(self) -> str:
        """Try to get account ID from API or user info."""
        # Try to list datasets to see if we can infer account ID
        # Or try to get user info
        try:
            # Try a simple API call to see error message
            url = f"{self.base_url}/v1/accounts/test/datasets"
            response = requests.get(url, headers=self.headers)
            # This will fail but might give us hints
        except:
            pass
        
        # For now, return empty and let user set it
        return ""
    
    def create_dataset(self, dataset_id: str, example_count: int) -> Dict[str, Any]:
        """Create a dataset."""
        url = f"{self.base_url}/v1/accounts/{self.account_id}/datasets"
        # datasetId must be at top level, not nested in dataset object
        payload = {
            "datasetId": dataset_id,
            "dataset": {
                "exampleCount": str(example_count),
            }
        }
        response = requests.post(url, headers=self.headers, json=payload)
        
        if response.status_code == 409:  # Conflict - already exists
            return self.get_dataset(dataset_id)
        
        if response.status_code != 200:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            try:
                error_json = response.json()
                if "message" in error_json:
                    error_msg = f"HTTP {response.status_code}: {error_json['message']}"
                elif "error" in error_json:
                    error_msg = f"HTTP {response.status_code}: {error_json['error']}"
            except:
                pass
            raise ValueError(f"Failed to create dataset: {error_msg}\nURL: {url}\nPayload: {json.dumps(payload, indent=2)}")
        
        return response.json()
    
    def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Get dataset info."""
        url = f"{self.base_url}/v1/accounts/{self.account_id}/datasets/{dataset_id}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def get_upload_endpoint(self, dataset_id: str, filename: str, file_size: int) -> Dict[str, Any]:
        """Get signed URL for uploading dataset file."""
        url = f"{self.base_url}/v1/accounts/{self.account_id}/datasets/{dataset_id}:getUploadEndpoint"
        response = requests.post(
            url,
            headers=self.headers,
            json={
                "filenameToSize": {
                    filename: str(file_size)
                }
            }
        )
        if response.status_code != 200:
            error_msg = response.text
            print(f"Error response: {error_msg}")
            # Check if dataset already exists and is ready
            try:
                dataset_info = self.get_dataset(dataset_id)
                if dataset_info.get("state") == "READY":
                    print("⚠ Dataset already exists and is in READY state - cannot upload new file")
                    raise ValueError(f"Dataset {dataset_id} is already in READY state. Delete it first or use a different dataset name.")
            except:
                pass
        response.raise_for_status()
        return response.json()
    
    def validate_upload(self, dataset_id: str) -> Dict[str, Any]:
        """Validate uploaded dataset."""
        url = f"{self.base_url}/v1/accounts/{self.account_id}/datasets/{dataset_id}:validateUpload"
        response = requests.post(url, headers=self.headers, json={})
        response.raise_for_status()
        return response.json()
    
    def create_sft_job(
        self,
        dataset_name: str,
        base_model: str,
        fine_tuned_model_id: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create supervised fine-tuning job."""
        url = f"{self.base_url}/v1/accounts/{self.account_id}/supervisedFineTuningJobs"
        payload = {
            "dataset": dataset_name,
            "baseModel": base_model,
            "outputModel": fine_tuned_model_id,  # Changed from fineTunedModelId to outputModel
            "displayName": f"QueryGPT SFT - {datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "learningRate": config["learning_rate"],
            "learningRateWarmupSteps": config["learning_rate_warmup_steps"],
            "epochs": config["epochs"],
            "loraRank": config["lora_rank"],
            "gradientAccumulationSteps": config["gradient_accumulation_steps"],
            "batchSize": config["batch_size"],
            "maxContextLength": config["max_context_length"],
        }
        response = requests.post(url, headers=self.headers, json=payload)
        
        if response.status_code != 200:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            try:
                error_json = response.json()
                if "message" in error_json:
                    error_msg = f"HTTP {response.status_code}: {error_json['message']}"
            except:
                pass
            raise ValueError(f"Failed to create fine-tuning job: {error_msg}\nURL: {url}\nPayload: {json.dumps(payload, indent=2)}")
        
        return response.json()
    
    def get_sft_job(self, job_id: str) -> Dict[str, Any]:
        """Get fine-tuning job status."""
        url = f"{self.base_url}/v1/accounts/{self.account_id}/supervisedFineTuningJobs/{job_id}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def list_sft_jobs(self, page_size: int = 10) -> List[Dict[str, Any]]:
        """List fine-tuning jobs."""
        url = f"{self.base_url}/v1/accounts/{self.account_id}/supervisedFineTuningJobs"
        response = requests.get(url, headers=self.headers, params={"pageSize": page_size})
        response.raise_for_status()
        return response.json().get("supervisedFineTuningJobs", [])
    
    def find_completed_job(self, display_name_pattern: str = None) -> Optional[Dict[str, Any]]:
        """Find a completed fine-tuning job, optionally matching display name."""
        jobs = self.list_sft_jobs(page_size=20)
        for job in jobs:
            state = job.get("state", "")
            if state in ["JOB_STATE_COMPLETED", "COMPLETED"]:
                if display_name_pattern:
                    display_name = job.get("displayName", "")
                    if display_name_pattern.lower() in display_name.lower():
                        return job
                else:
                    return job  # Return first completed job
        return None


def create_and_upload_dataset(client: FireworksAPIClient, dataset_id: str, file_path: str) -> Dict[str, Any]:
    """Create a dataset and upload the JSONL file to Fireworks."""
    print(f"\nCreating dataset '{dataset_id}'...")
    
    # Count examples in JSONL file
    example_count = 0
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                example_count += 1
    
    print(f"  Found {example_count} examples in {file_path}")
    
    # Check if dataset already exists
    try:
        existing_dataset = client.get_dataset(dataset_id)
        existing_count = existing_dataset.get("exampleCount", 0)
        
        if existing_dataset.get("state") == "READY":
            if existing_count == example_count:
                print(f"✓ Dataset '{dataset_id}' already exists with {example_count} examples and is READY")
                print("  Using existing dataset (no upload needed)")
                return existing_dataset
            else:
                print(f"⚠ Dataset '{dataset_id}' exists but has {existing_count} examples (need {example_count})")
                print(f"  Creating new dataset with suffix '-v2'...")
                dataset_id = f"{dataset_id}-v2"
                dataset = client.create_dataset(dataset_id, example_count)
                print(f"✓ Dataset '{dataset_id}' created")
        else:
            print(f"⚠ Dataset '{dataset_id}' exists but is not READY (state: {existing_dataset.get('state')})")
            print("  Will attempt to upload to existing dataset...")
            dataset = existing_dataset
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:  # Not found - create new
            dataset = client.create_dataset(dataset_id, example_count)
            print(f"✓ Dataset '{dataset_id}' created")
        elif e.response.status_code == 409:  # Conflict - already exists
            print(f"⚠ Dataset '{dataset_id}' already exists, using existing dataset")
            dataset = client.get_dataset(dataset_id)
        else:
            raise
    
    # Upload the file
    print(f"\nUploading {file_path}...")
    file_size = os.path.getsize(file_path)
    print(f"  File size: {file_size / 1024 / 1024:.2f} MB")
    
    upload_endpoint = client.get_upload_endpoint(dataset_id, os.path.basename(file_path), file_size)
    
    if "filenameToSignedUrls" not in upload_endpoint or not upload_endpoint["filenameToSignedUrls"]:
        raise ValueError("Failed to get upload endpoint URLs")
    
    signed_url = upload_endpoint["filenameToSignedUrls"].get(os.path.basename(file_path))
    if signed_url is None:
        raise ValueError(f"Failed to get signed URL for file: {file_path}")
    
    with open(file_path, "rb") as f:
        response = requests.put(
            signed_url,
            data=f.read(),
            headers={
                "Content-Type": "application/octet-stream",
                "x-goog-content-length-range": f"{file_size},{file_size}",
            },
            timeout=300.0,  # 5 minute timeout for large files
        )
        response.raise_for_status()
    
    print("✓ File uploaded")
    
    # Validate the upload
    print("\nValidating upload...")
    client.validate_upload(dataset_id)
    print("✓ Upload validated")
    
    # Wait for dataset to be READY
    print("\nWaiting for dataset to be ready...")
    max_wait_time = 300  # 5 minutes max
    check_interval = 5
    elapsed = 0
    
    while elapsed < max_wait_time:
        dataset = client.get_dataset(dataset_id)
        state = dataset.get("state", "")
        if state == "READY":
            print("✓ Dataset is ready")
            break
        elif state in ["FAILED", "ERROR"]:
            raise RuntimeError(f"Dataset validation failed with state: {state}")
        print(f"  Dataset state: {state} (waiting {check_interval}s...)")
        time.sleep(check_interval)
        elapsed += check_interval
    
    if dataset.get("state") != "READY":
        raise RuntimeError(f"Dataset did not become READY within {max_wait_time}s. Current state: {dataset.get('state')}")
    
    return dataset


def create_fine_tuning_job(
    client: FireworksAPIClient,
    train_dataset: Dict[str, Any],
    base_model: str,
    fine_tuned_model_id: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Create a supervised fine-tuning job."""
    print("\n" + "="*80)
    print("STEP 3: CREATE FINE-TUNING JOB")
    print("="*80)
    
    dataset_name = train_dataset.get("name") or train_dataset.get("datasetId")
    if dataset_name is None:
        raise ValueError("Train dataset name is None")
    
    # Check dataset is READY
    dataset_state = train_dataset.get("state", "")
    if dataset_state != "READY":
        raise ValueError(f"Dataset must be in READY state, but it's in {dataset_state} state. Please wait for dataset validation to complete.")
    
    # Format base model: convert fireworks/model-name to accounts/fireworks/models/model-name
    if base_model.startswith("fireworks/"):
        base_model_formatted = f"accounts/fireworks/models/{base_model.replace('fireworks/', '')}"
    elif not base_model.startswith("accounts/"):
        base_model_formatted = f"accounts/fireworks/models/{base_model}"
    else:
        base_model_formatted = base_model
    
    # Format outputModel as accounts/<account-id>/models/<model-id>
    if not fine_tuned_model_id.startswith("accounts/"):
        output_model = f"accounts/{client.account_id}/models/{fine_tuned_model_id}"
    else:
        output_model = fine_tuned_model_id
    
    print(f"\nBase Model: {base_model_formatted}")
    print(f"Fine-tuned Model ID: {output_model}")
    print(f"Training Dataset: {dataset_name}")
    print(f"\nFine-tuning Configuration:")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  LoRA Rank: {config['lora_rank']}")
    print(f"  Batch Size: {config['batch_size']}")
    
    try:
        sftj = client.create_sft_job(
            dataset_name=dataset_name,
            base_model=base_model_formatted,
            fine_tuned_model_id=output_model,
            config=config
        )
    except ValueError as e:
        error_msg = str(e)
        if "not tunable" in error_msg.lower() or "missing conversation config" in error_msg.lower():
            print(f"\n❌ Error: The model '{base_model_formatted}' does not support fine-tuning.")
            print("\nSuggested tunable models:")
            print("  - accounts/fireworks/models/qwen2.5-7b-instruct")
            print("  - accounts/fireworks/models/qwen3-30b-a3b-instruct-2507")
            print("  - accounts/fireworks/models/llama-v3p3-8b-instruct")
            print("\nYou can specify a different model with:")
            print("  python3 run_sft_workflow.py --base-model accounts/fireworks/models/qwen2.5-7b-instruct")
        raise
    
    sftj_name = sftj.get("name") or sftj.get("jobId")
    if sftj_name is None:
        raise ValueError("SFTJ name is None")
    
    sftj_id = sftj_name.split("/")[-1] if "/" in str(sftj_name) else str(sftj_name)
    print(f"\n✓ Fine-tuning job created: {sftj_id}")
    print(f"\nMonitor progress at:")
    print(f"  https://app.fireworks.ai/dashboard/fine-tuning/supervised/{sftj_id}")
    
    return sftj


def monitor_fine_tuning_job(client: FireworksAPIClient, sftj_id: str, check_interval: int = 30) -> Dict[str, Any]:
    """Monitor fine-tuning job until completion."""
    print("\n" + "="*80)
    print("STEP 4: MONITOR FINE-TUNING JOB")
    print("="*80)
    
    print(f"\nMonitoring job {sftj_id}...")
    print("(This may take 10-60 minutes depending on dataset size)")
    
    last_state = None
    start_time = time.time()
    
    while True:
        sftj = client.get_sft_job(sftj_id)
        current_state = sftj.get("state") or sftj.get("status")
        
        if current_state != last_state:
            elapsed = time.time() - start_time
            print(f"\n[{elapsed/60:.1f}m] State: {current_state}")
            last_state = current_state
        
        if current_state in ["JOB_STATE_COMPLETED", "COMPLETED", "completed"]:
            elapsed = time.time() - start_time
            print(f"\n✓ Fine-tuning completed in {elapsed/60:.1f} minutes")
            return sftj
        elif current_state in ["JOB_STATE_FAILED", "FAILED", "failed", "JOB_STATE_CANCELLED", "CANCELLED", "cancelled"]:
            raise RuntimeError(f"Fine-tuning job {current_state.lower()}")
        
        time.sleep(check_interval)


def deploy_fine_tuned_model(client: FireworksAPIClient, fine_tuned_model_id: str) -> str:
    """Deploy the fine-tuned model for inference."""
    print("\n" + "="*80)
    print("STEP 5: DEPLOY FINE-TUNED MODEL")
    print("="*80)
    
    print(f"\nDeploying model: {fine_tuned_model_id}")
    print("(Note: You may need to deploy manually via UI or use deployment API)")
    print(f"\nModel will be available at: accounts/fireworks/models/{fine_tuned_model_id}")
    
    # For now, just return the model ID
    # Actual deployment might require additional steps or manual action
    return fine_tuned_model_id


def run_post_tuning_evaluation(fine_tuned_model_id: str, base_model_for_agents: str = None) -> Dict[str, Any]:
    """Run evaluation after fine-tuning."""
    print("\n" + "="*80)
    print("STEP 6: POST-TUNING EVALUATION")
    print("="*80)
    
    # Load evaluation data
    print(f"\nLoading evaluation data from {EVALUATION_DATA_FILE}...")
    with open(EVALUATION_DATA_FILE) as f:
        golden_data = json.load(f)
    print(f"✓ Loaded {len(golden_data)} test cases")
    
    # Format fine-tuned model ID
    if fine_tuned_model_id.startswith("accounts/"):
        fine_tuned_model = fine_tuned_model_id
    else:
        fine_tuned_model = f"accounts/fireworks/models/{fine_tuned_model_id}"
    
    # Use fine-tuned model for evaluation
    # Note: The fine-tuned model needs to be deployed before it can be used
    print(f"\nUsing fine-tuned model for evaluation: {fine_tuned_model}")
    print("Note: If you get an error, the model may need to be deployed first.")
    print("Check the Fireworks dashboard to deploy the model, or it may auto-deploy.")
    evaluation_model = fine_tuned_model
    
    print(f"\nInitializing QueryGPT with model: {evaluation_model}")
    querygpt = FireworksQueryGPT(model=evaluation_model, intent_model=evaluation_model)
    print("✓ QueryGPT initialized")
    
    # Load database
    print("\nLoading database...")
    db_conn = load_db()
    print("✓ Database loaded")
    
    # Run evaluation
    print("\nRunning evaluation...")
    evaluator = AgentEvaluator(querygpt, db_conn, golden_data)
    summary = evaluator.evaluate_all(verbose=False)
    
    # Calculate reward metrics
    evaluation_results = {
        "summary": summary,
        "detailed_results": evaluator.results,
        "agent_metrics": querygpt.get_metrics_summary(),
    }
    
    reward_metrics = calculate_reward_metrics(evaluation_results)
    evaluation_results["reward_metrics"] = reward_metrics
    
    # Save post-tuning results
    post_tuning_file = "evaluation_results_post_tuning.json"
    with open(post_tuning_file, "w") as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"\n✓ Post-tuning evaluation saved to {post_tuning_file}")
    
    # Print summary
    print("\n" + "-"*80)
    print("POST-TUNING METRICS:")
    print("-"*80)
    print(f"Overall Accuracy: {summary['overall']['accuracy']*100:.1f}%")
    print(f"Average Latency: {reward_metrics['avg_latency_ms']:.1f}ms")
    print(f"Total Tokens: {reward_metrics['total_tokens']:,}")
    print(f"Estimated Cost: ${reward_metrics['estimated_cost']:.4f}")
    print(f"Reward Score: {reward_metrics['reward_score']:.3f}")
    
    db_conn.close()
    return evaluation_results


def compare_results(baseline: Dict[str, Any], post_tuning: Dict[str, Any]) -> Dict[str, Any]:
    """Compare baseline and post-tuning results."""
    print("\n" + "="*80)
    print("STEP 7: COMPARISON REPORT")
    print("="*80)
    
    baseline_reward = baseline["reward_metrics"]
    post_tuning_reward = post_tuning["reward_metrics"]
    
    comparison = {
        "accuracy": {
            "baseline": baseline_reward["accuracy_score"],
            "post_tuning": post_tuning_reward["accuracy_score"],
            "improvement": post_tuning_reward["accuracy_score"] - baseline_reward["accuracy_score"],
            "improvement_pct": ((post_tuning_reward["accuracy_score"] - baseline_reward["accuracy_score"]) / baseline_reward["accuracy_score"] * 100) if baseline_reward["accuracy_score"] > 0 else 0,
        },
        "latency": {
            "baseline_ms": baseline_reward["avg_latency_ms"],
            "post_tuning_ms": post_tuning_reward["avg_latency_ms"],
            "improvement_ms": baseline_reward["avg_latency_ms"] - post_tuning_reward["avg_latency_ms"],
            "improvement_pct": ((baseline_reward["avg_latency_ms"] - post_tuning_reward["avg_latency_ms"]) / baseline_reward["avg_latency_ms"] * 100) if baseline_reward["avg_latency_ms"] > 0 else 0,
        },
        "cost": {
            "baseline": baseline_reward["estimated_cost"],
            "post_tuning": post_tuning_reward["estimated_cost"],
            "improvement": baseline_reward["estimated_cost"] - post_tuning_reward["estimated_cost"],
            "improvement_pct": ((baseline_reward["estimated_cost"] - post_tuning_reward["estimated_cost"]) / baseline_reward["estimated_cost"] * 100) if baseline_reward["estimated_cost"] > 0 else 0,
        },
        "reward_score": {
            "baseline": baseline_reward["reward_score"],
            "post_tuning": post_tuning_reward["reward_score"],
            "improvement": post_tuning_reward["reward_score"] - baseline_reward["reward_score"],
            "improvement_pct": ((post_tuning_reward["reward_score"] - baseline_reward["reward_score"]) / baseline_reward["reward_score"] * 100) if baseline_reward["reward_score"] > 0 else 0,
        },
    }
    
    # Print comparison
    print("\n" + "-"*80)
    print("METRICS COMPARISON:")
    print("-"*80)
    
    print(f"\n📊 ACCURACY:")
    print(f"  Baseline:    {baseline_reward['accuracy_score']*100:.2f}%")
    print(f"  Post-tuning: {post_tuning_reward['accuracy_score']*100:.2f}%")
    improvement = comparison["accuracy"]["improvement_pct"]
    print(f"  Change:      {improvement:+.2f}% {'✅' if improvement > 0 else '❌'}")
    
    print(f"\n⚡ LATENCY:")
    print(f"  Baseline:    {baseline_reward['avg_latency_ms']:.1f}ms")
    print(f"  Post-tuning: {post_tuning_reward['avg_latency_ms']:.1f}ms")
    improvement = comparison["latency"]["improvement_pct"]
    print(f"  Change:      {improvement:+.2f}% {'✅' if improvement > 0 else '❌'}")
    
    print(f"\n💰 COST:")
    print(f"  Baseline:    ${baseline_reward['estimated_cost']:.4f}")
    print(f"  Post-tuning: ${post_tuning_reward['estimated_cost']:.4f}")
    improvement = comparison["cost"]["improvement_pct"]
    print(f"  Change:      {improvement:+.2f}% {'✅' if improvement < 0 else '❌'}")
    
    print(f"\n🎯 REWARD SCORE:")
    print(f"  Baseline:    {baseline_reward['reward_score']:.3f}")
    print(f"  Post-tuning: {post_tuning_reward['reward_score']:.3f}")
    improvement = comparison["reward_score"]["improvement"]
    print(f"  Change:      {improvement:+.3f} {'✅' if improvement > 0 else '❌'}")
    
    # Save comparison
    comparison_file = "sft_comparison_report.json"
    with open(comparison_file, "w") as f:
        json.dump({
            "baseline": baseline,
            "post_tuning": post_tuning,
            "comparison": comparison,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\n✓ Comparison report saved to {comparison_file}")
    
    return comparison


def main():
    """Main workflow execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run complete SFT workflow on Fireworks AI")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline evaluation")
    parser.add_argument("--skip-upload", action="store_true", help="Skip dataset upload (use existing)")
    parser.add_argument("--skip-tuning", action="store_true", help="Skip fine-tuning (use existing model)")
    parser.add_argument("--model-id", type=str, help="Fine-tuned model ID (if skipping tuning)")
    parser.add_argument("--dataset-file", type=str, default=GOLDEN_DATASET_FILE, help="Path to JSONL dataset file")
    parser.add_argument("--base-model", type=str, default=BASE_MODEL, help="Base model to fine-tune")
    
    args = parser.parse_args()
    
    # Check API key
    if not os.getenv("FIREWORKS_API_KEY"):
        print("Error: FIREWORKS_API_KEY not found in environment")
        print("Set it in .env file or export it")
        return
    
    # Check account ID (may be needed for some operations)
    account_id = os.getenv("FIREWORKS_ACCOUNT_ID")
    if not account_id:
        print("⚠ Warning: FIREWORKS_ACCOUNT_ID not set. Some operations may fail.")
    
    # Initialize Fireworks API client
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        print("Error: FIREWORKS_API_KEY not found in environment")
        return
    
    account_id = os.getenv("FIREWORKS_ACCOUNT_ID")
    if not account_id:
        print("\n⚠ FIREWORKS_ACCOUNT_ID not set.")
        print("You can find your account ID by:")
        print("  1. Check the Fireworks dashboard URL")
        print("  2. Or check your API settings in the dashboard")
        print("  3. Or try using 'fireworks' as the account ID (for some accounts)")
        print("\nSetting it in .env file:")
        print("  FIREWORKS_ACCOUNT_ID=your-account-id-here")
        print("\nAttempting to continue...")
    
    client = FireworksAPIClient(api_key, account_id)
    
    if not client.account_id:
        print("\n❌ Error: Could not determine account ID. Please set FIREWORKS_ACCOUNT_ID in .env")
        return
    
    try:
        # Step 1: Baseline Evaluation
        if not args.skip_baseline:
            baseline_results = run_baseline_evaluation()
        else:
            print("\n⚠ Skipping baseline evaluation")
            if Path("evaluation_results_baseline.json").exists():
                with open("evaluation_results_baseline.json") as f:
                    baseline_results = json.load(f)
                print("✓ Loaded existing baseline results")
            else:
                print("❌ No baseline results found. Run without --skip-baseline first.")
                return
        
        # Step 2: Upload Dataset
        if not args.skip_upload:
            print("\n" + "="*80)
            print("STEP 2: UPLOAD DATASET")
            print("="*80)
            
            if not Path(args.dataset_file).exists():
                print(f"❌ Error: Dataset file not found: {args.dataset_file}")
                return
            
            train_dataset = create_and_upload_dataset(client, DATASET_ID, args.dataset_file)
        else:
            print("\n⚠ Skipping dataset upload")
            train_dataset = client.get_dataset(DATASET_ID)
            print(f"✓ Using existing dataset: {DATASET_ID}")
        
        # Step 3: Create Fine-Tuning Job
        if not args.skip_tuning:
            fine_tuned_model_id = f"{FINE_TUNED_MODEL_PREFIX}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            sftj = create_fine_tuning_job(client, train_dataset, args.base_model, fine_tuned_model_id, SFT_CONFIG)
            
            # Step 4: Monitor Job
            sftj_name = sftj.get("name") or sftj.get("jobId")
            sftj_id = sftj_name.split("/")[-1] if "/" in str(sftj_name) else str(sftj_name)
            sftj = monitor_fine_tuning_job(client, sftj_id)
            
            # Get the actual output model ID from the completed job
            completed_job = client.get_sft_job(sftj_id)
            output_model = completed_job.get("outputModel")
            if output_model:
                fine_tuned_model_id = output_model
                print(f"\n✓ Fine-tuned model ID: {fine_tuned_model_id}")
            else:
                # Fallback to constructed ID
                fine_tuned_model_id = f"accounts/{client.account_id}/models/{fine_tuned_model_id}"
            
            # Step 5: Deploy Model (may require manual steps)
            fine_tuned_model_id = deploy_fine_tuned_model(client, fine_tuned_model_id)
        else:
            if args.model_id:
                fine_tuned_model_id = args.model_id
            else:
                # Try to find a completed job automatically
                print("\n⚠ No model ID provided. Searching for completed fine-tuning job...")
                # First try QueryGPT job, then any completed job
                completed_job = client.find_completed_job(display_name_pattern="QueryGPT")
                if not completed_job:
                    completed_job = client.find_completed_job()  # Get any completed job
                
                if completed_job:
                    fine_tuned_model_id = completed_job.get("outputModel")
                    if fine_tuned_model_id:
                        print(f"✓ Found completed job: {completed_job.get('displayName')}")
                        print(f"✓ Using fine-tuned model: {fine_tuned_model_id}")
                    else:
                        print("❌ Error: Completed job found but no outputModel")
                        print("   Please provide --model-id manually")
                        return
                else:
                    print("❌ Error: No completed fine-tuning job found.")
                    print("   Please provide --model-id or complete a fine-tuning job first.")
                    print("\n   Example:")
                    print("   python3 run_sft_workflow.py --skip-baseline --skip-upload --skip-tuning \\")
                    print("     --model-id accounts/sanatmouli-clqab3ddm/models/your-model-id")
                    return
            print(f"\n⚠ Skipping fine-tuning, using model: {fine_tuned_model_id}")
        
        # Step 6: Post-Tuning Evaluation
        # Use the fine-tuned model for all agents to test improvements
        print(f"\n⚠ Note: Fine-tuned model will be used for evaluation once it's deployed.")
        print(f"   If the model isn't available yet, you may need to deploy it manually or wait.")
        post_tuning_results = run_post_tuning_evaluation(fine_tuned_model_id)
        
        # Step 7: Compare Results
        comparison = compare_results(baseline_results, post_tuning_results)
        
        print("\n" + "="*80)
        print("✅ SFT WORKFLOW COMPLETED")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Workflow interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

