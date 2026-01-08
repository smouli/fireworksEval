# SFT Workflow Guide

Complete guide for running Supervised Fine-Tuning (SFT) on Fireworks AI with before/after evaluation.

## Overview

This workflow automates the entire SFT process:
1. **Baseline Evaluation** - Test model performance before fine-tuning
2. **Dataset Upload** - Upload your training dataset to Fireworks
3. **Fine-Tuning Job** - Create and monitor the SFT job
4. **Model Deployment** - Deploy the fine-tuned model
5. **Post-Tuning Evaluation** - Test model performance after fine-tuning
6. **Comparison Report** - Compare metrics (accuracy, latency, cost)

## Prerequisites

1. **Install Fireworks SDK:**
   ```bash
   pip install fireworks
   # Or with all dependencies:
   pip install fireworks[all]
   ```

2. **Set Environment Variables:**
   Create or update your `.env` file:
   ```bash
   FIREWORKS_API_KEY=your_api_key_here
   FIREWORKS_ACCOUNT_ID=your_account_id_here  # Optional but recommended
   FIREWORKS_MODEL=accounts/fireworks/models/qwen3-30b-a3b-instruct-2507
   ```

3. **Prepare Dataset:**
   - Your dataset should be in JSONL format (already created: `golden_dataset_sft.jsonl`)
   - Each line should be a JSON object with a `messages` array
   - Format: `{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`

## Usage

### Full Workflow (Recommended)

Run the complete workflow from start to finish:

```bash
python3 run_sft_workflow.py
```

This will:
- Run baseline evaluation
- Upload your dataset
- Create fine-tuning job
- Monitor until completion
- Deploy model
- Run post-tuning evaluation
- Generate comparison report

### Partial Workflows

**Skip baseline evaluation** (if you already have baseline results):
```bash
python3 run_sft_workflow.py --skip-baseline
```

**Skip dataset upload** (if dataset already exists):
```bash
python3 run_sft_workflow.py --skip-upload
```

**Skip fine-tuning** (if you already have a fine-tuned model):
```bash
python3 run_sft_workflow.py --skip-tuning --model-id your-model-id
```

**Use custom dataset file:**
```bash
python3 run_sft_workflow.py --dataset-file custom_dataset.jsonl
```

**Use different base model:**
```bash
python3 run_sft_workflow.py --base-model accounts/fireworks/models/your-model
```

## Reward Function

The workflow calculates a composite reward score based on:

1. **Accuracy (60% weight)**
   - Intent agent accuracy
   - Table agent F1 score
   - Column prune agent F1 score
   - SQL generation similarity
   - Overall accuracy

2. **Latency (25% weight)**
   - Average latency across all agents
   - Normalized: lower latency = higher score

3. **Cost (15% weight)**
   - Estimated cost based on token usage
   - Normalized: lower cost = higher score

**Reward Score Formula:**
```
reward_score = (accuracy * 0.60) + (latency_score * 0.25) + (cost_score * 0.15)
```

## Output Files

The workflow generates several output files:

1. **`evaluation_results_baseline.json`** - Baseline evaluation results
2. **`evaluation_results_post_tuning.json`** - Post-tuning evaluation results
3. **`sft_comparison_report.json`** - Comparison of before/after metrics

## Fine-Tuning Configuration

Default configuration (can be modified in script):
- **Learning Rate:** 1e-5
- **Epochs:** 1
- **LoRA Rank:** 16
- **Batch Size:** 16384
- **Max Context Length:** 16384
- **Gradient Accumulation Steps:** 2
- **Learning Rate Warmup Steps:** 200

## Monitoring

During fine-tuning, you can monitor progress:
1. **Via Script:** The script will print status updates every 30 seconds
2. **Via Dashboard:** Visit the URL printed when the job is created:
   ```
   https://app.fireworks.ai/dashboard/fine-tuning/supervised/{job_id}
   ```

## Expected Timeline

- **Baseline Evaluation:** 5-15 minutes (depending on dataset size)
- **Dataset Upload:** 1-5 minutes (depending on file size)
- **Fine-Tuning:** 10-60 minutes (depending on dataset size and model)
- **Post-Tuning Evaluation:** 5-15 minutes
- **Total:** ~30-90 minutes

## Troubleshooting

### Error: "Fireworks SDK not installed"
```bash
pip install fireworks
```

### Error: "FIREWORKS_API_KEY not found"
Make sure your `.env` file contains:
```
FIREWORKS_API_KEY=your_key_here
```

### Error: "Dataset file not found"
Make sure `golden_dataset_sft.jsonl` exists, or specify with `--dataset-file`

### Fine-tuning job fails
- Check the job status in the Fireworks dashboard
- Verify your dataset format is correct (JSONL with messages array)
- Ensure you have sufficient quota/credits

### Model deployment issues
- Some models may require manual deployment via the UI
- Check the Fireworks dashboard for deployment options
- Ensure the fine-tuned model ID is correct

## Next Steps

After completing the workflow:

1. **Review Comparison Report:**
   ```bash
   cat sft_comparison_report.json | python3 -m json.tool
   ```

2. **Analyze Improvements:**
   - Check accuracy improvements
   - Review latency changes
   - Assess cost impact

3. **Iterate:**
   - If results are good, consider expanding the dataset
   - If results are poor, try adjusting hyperparameters
   - Consider RFT (Reinforcement Fine-Tuning) for further improvements

## Cost Estimation

The workflow estimates costs based on:
- Token usage during evaluation
- Fine-tuning job costs (check Fireworks pricing)
- Inference costs for fine-tuned model

**Note:** Actual costs may vary. Check Fireworks pricing page for accurate rates.

## Support

For issues or questions:
1. Check Fireworks AI documentation: https://docs.fireworks.ai
2. Review the comparison report for detailed metrics
3. Check job logs in the Fireworks dashboard

