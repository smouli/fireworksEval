# Fireworks QueryGPT Implementation

Complete QueryGPT-style text-to-SQL system using Fireworks AI with function calling for structured outputs.

## Quick Start

### 1. Set up API Key and Models

```bash
export FIREWORKS_API_KEY='your-api-key-here'
# Optional: Set model names (check https://app.fireworks.ai/models for available models)
export FIREWORKS_MODEL='accounts/fireworks/models/firefunction-v2'
export FIREWORKS_INTENT_MODEL='accounts/fireworks/models/llama-v3-8b-instruct'
```

**Important**: Check your Fireworks dashboard (https://app.fireworks.ai/models) to see which models are available to your account. Some models may need to be deployed first.

See `MODEL_SETUP.md` for detailed instructions.

### 2. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv_querygpt
source venv_querygpt/bin/activate
pip install openai

# Or use the setup script
./setup_querygpt.sh
```

### 3. Run Example

```bash
python3 example_querygpt_usage.py
```

### 4. Evaluate Performance

```bash
# Evaluate on 5 queries
python3 evaluate_querygpt.py 5

# Evaluate on all queries
python3 evaluate_querygpt.py
```

## Architecture

The system implements 4 agents using Fireworks AI:

1. **Intent Agent** - Classifies question into workspace (mobility, customer_analytics, etc.)
2. **Table Agent** - Identifies relevant tables for the query
3. **Column Prune Agent** - Reduces schema size by pruning irrelevant columns
4. **SQL Generation Agent** - Generates SQL using few-shot learning

## Fireworks Optimizations

### API Features (Built-in)
- ✅ **Function Calling** - Structured outputs via function schemas
- ✅ **Fast Models** - Optimized llama-v3p3 models
- ✅ **Model Selection** - Different models for different tasks (8B for intent, 70B for SQL)
- ✅ **Temperature Control** - Fine-tuned per agent
- ✅ **Token Tracking** - Built-in usage tracking

### Code Optimizations (Our Implementation)
- ✅ **Column Pruning** - Reduces tokens by 40-60%
- ✅ **Few-Shot Learning** - Workspace-based SQL samples
- ✅ **Workspace RAG** - Intent-based example selection
- ⏳ **Parallel Calls** - Potential for async optimization
- ⏳ **Streaming** - Potential for faster UX

See `FIREWORKS_OPTIMIZATIONS.md` for detailed explanation.

## Usage

### Basic Usage

```python
from fireworks_querygpt import FireworksQueryGPT

# Initialize
querygpt = FireworksQueryGPT()

# Generate SQL
result = querygpt.generate_sql("How many trips were completed yesterday?")

print(result["sql"])
print(result["explanation"])
print(f"Latency: {result['total_latency_ms']}ms")
```

### Individual Agents

```python
# Intent Agent
intent = querygpt.intent_agent("Show me drivers in Seattle")
print(intent["workspace_id"])  # "mobility"
print(intent["confidence"])     # 0.95

# Table Agent
tables = querygpt.table_agent("How many trips?", "mobility")
print(tables["relevant_tables"])  # ["trips", "cities"]

# Column Prune Agent
schema = workspace["table_schemas"]["trips"]
pruned = querygpt.column_prune_agent("Count trips", "trips", schema)
print(pruned["token_savings_pct"])  # 45.2

# SQL Generation
sql = querygpt.sql_generation_agent(
    question="Count trips",
    workspace_id="mobility",
    relevant_tables=["trips"],
    pruned_schemas={"trips": pruned["pruned_schema"]},
    sql_samples=workspace["sql_samples"]
)
```

### Metrics

```python
# Get metrics summary
metrics = querygpt.get_metrics_summary()

print(metrics["intent_agent"]["success_rate"])      # 1.0
print(metrics["intent_agent"]["avg_latency_ms"])    # 245.3
print(metrics["table_agent"]["avg_tokens"])         # 1250
```

## Evaluation

The evaluation script measures:

- **Tool Calling Success Rate** - % of successful function calls per agent
- **Table Selection Accuracy** - Precision, recall, F1 for table selection
- **Column Prune Savings** - Average token reduction percentage
- **SQL Execution Rate** - % of generated SQL that executes successfully
- **SQL Correctness** - Comparison with expected results
- **Latency** - Average pipeline latency

```bash
python3 evaluate_querygpt.py 10  # Evaluate on 10 queries
```

Results are saved to `evaluation_results.json`.

## Configuration

### Model Selection

```python
querygpt = FireworksQueryGPT(
    model="accounts/fireworks/models/llama-v3p3-70b-instruct",  # Main model
    intent_model="accounts/fireworks/models/llama-v3p3-8b-instruct",  # Intent model
)
```

### Custom Base URL

```python
querygpt = FireworksQueryGPT(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key="your-key"
)
```

## Performance Tips

1. **Use smaller models for simple tasks** - Intent agent uses 8B model
2. **Column pruning saves tokens** - Typically 40-60% reduction
3. **Workspace selection improves accuracy** - Narrower search space
4. **Few-shot examples help** - 5 examples per workspace

## Files

- `fireworks_querygpt.py` - Main QueryGPT implementation
- `evaluate_querygpt.py` - Evaluation script
- `example_querygpt_usage.py` - Example usage
- `FIREWORKS_OPTIMIZATIONS.md` - Detailed optimization explanation
- `querygpt_workspaces/` - Workspace data and SQL samples

## Requirements

- Python 3.11+
- `openai` package (for Fireworks API)
- `FIREWORKS_API_KEY` environment variable

## Next Steps

1. Implement parallel column pruning with `asyncio`
2. Add streaming for faster perceived latency
3. Implement caching for repeated queries
4. Fine-tune models on SQL generation

