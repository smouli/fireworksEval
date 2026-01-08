# FireworksAI Text-to-SQL Evaluation Project

A comprehensive evaluation framework for text-to-SQL systems using Fireworks AI, implementing a QueryGPT-style multi-agent architecture with workspace-based RAG, column pruning, and few-shot learning.

## Overview

This project implements and evaluates a production-ready text-to-SQL system that:
- Uses a 4-agent pipeline (Intent → Table Selection → Column Pruning → SQL Generation)
- Leverages Fireworks AI function calling for structured outputs
- Implements workspace-based RAG to reduce hallucination
- Optimizes token usage through column pruning (40-60% reduction)
- Supports fine-tuning workflows for improved accuracy

## Features

- **Multi-Agent Architecture**: Intent classification, table selection, column pruning, and SQL generation
- **Workspace-Based RAG**: Domain-specific workspaces (Mobility, Customer Analytics, Vehicle Operations, Promotions)
- **Column Pruning**: Reduces token usage by 40-60% by selecting only relevant columns
- **Function Calling**: Structured outputs using Fireworks AI tool calling
- **Fine-Tuning Support**: Complete SFT workflow with before/after evaluation
- **Comprehensive Evaluation**: Metrics for accuracy, latency, cost, and tool calling performance
- **Prompt Caching Optimization**: Structured prompts for maximum cache hits

## Setup

### Prerequisites

- Python 3.8+
- Fireworks AI API key
- SQLite database (created automatically)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd fireworksEval

# Create virtual environment
python3 -m venv venv_querygpt
source venv_querygpt/bin/activate  # On Windows: venv_querygpt\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# Or install manually:
pip install openai python-dotenv requests sqlite3

# Set up environment variables
cp .env.example .env  # If exists, or create .env file
# Add your API key:
# FIREWORKS_API_KEY=your-api-key-here
# FIREWORKS_MODEL=accounts/fireworks/models/firefunction-v2
# FIREWORKS_INTENT_MODEL=accounts/fireworks/models/llama-v3-8b-instruct
```

### Database Setup

The database is automatically created when you run the scripts. To manually create it:

```bash
python3 create_dummy_db.py
```

This creates a SQLite database (`Chinook.db`) with tables for trips, drivers, vehicles, cities, customers, and promotions.

## Database Schema

The database contains 7 tables modeling a ride-sharing/mobility platform:

### Core Tables

**`trips`** - Trip transactions
- `trip_id` (INTEGER, PK)
- `driver_id` (INTEGER, FK → drivers.driver_id)
- `customer_id` (INTEGER, FK → customers.customer_id)
- `vehicle_id` (INTEGER, FK → vehicles.vehicle_id)
- `pickup_city_id` (INTEGER, FK → cities.city_id)
- `dropoff_city_id` (INTEGER, FK → cities.city_id)
- `trip_status` (TEXT, NOT NULL) - e.g., 'completed', 'cancelled', 'in_progress'
- `trip_type` (TEXT)
- `distance_miles` (REAL)
- `duration_minutes` (INTEGER)
- `fare_amount` (REAL, NOT NULL)
- `tip_amount` (REAL, DEFAULT 0.0)
- `total_amount` (REAL, NOT NULL)
- `pickup_time` (TIMESTAMP, NOT NULL)
- `dropoff_time` (TIMESTAMP)
- `payment_method` (TEXT)
- `rating` (INTEGER)

**`drivers`** - Driver information
- `driver_id` (INTEGER, PK)
- `first_name` (TEXT, NOT NULL)
- `last_name` (TEXT, NOT NULL)
- `email` (TEXT, NOT NULL, UNIQUE)
- `phone` (TEXT)
- `city_id` (INTEGER, FK → cities.city_id)
- `status` (TEXT, NOT NULL) - e.g., 'active', 'inactive'
- `rating` (REAL)
- `total_trips` (INTEGER, DEFAULT 0)
- `joined_date` (DATE)
- `vehicle_type` (TEXT)

**`vehicles`** - Vehicle information
- `vehicle_id` (INTEGER, PK)
- `driver_id` (INTEGER, FK → drivers.driver_id, NOT NULL)
- `make` (TEXT, NOT NULL) - e.g., 'Tesla', 'Toyota'
- `model` (TEXT, NOT NULL)
- `year` (INTEGER)
- `color` (TEXT)
- `license_plate` (TEXT, UNIQUE)
- `vehicle_type` (TEXT) - e.g., 'sedan', 'suv'
- `status` (TEXT)

**`customers`** - Customer information
- `customer_id` (INTEGER, PK)
- `first_name` (TEXT, NOT NULL)
- `last_name` (TEXT, NOT NULL)
- `email` (TEXT, NOT NULL, UNIQUE)
- `phone` (TEXT)
- `city_id` (INTEGER, FK → cities.city_id)
- `signup_date` (DATE)
- `total_rides` (INTEGER, DEFAULT 0)
- `lifetime_value` (REAL, DEFAULT 0.0)

**`cities`** - City/location information
- `city_id` (INTEGER, PK)
- `city_name` (TEXT, NOT NULL)
- `state` (TEXT)
- `country` (TEXT, NOT NULL)
- `timezone` (TEXT)
- `population` (INTEGER)
- `created_at` (TIMESTAMP, DEFAULT CURRENT_TIMESTAMP)

**`promotions`** - Marketing promotions
- `promotion_id` (INTEGER, PK)
- `promotion_code` (TEXT, NOT NULL)
- `description` (TEXT)
- `discount_type` (TEXT)
- `discount_value` (REAL)
- `start_date` (DATE, NOT NULL)
- `end_date` (DATE)
- `max_uses` (INTEGER)
- `current_uses` (INTEGER, DEFAULT 0)
- `is_active` (INTEGER, DEFAULT 1)

**`trip_promotions`** - Junction table linking trips to promotions
- `trip_id` (INTEGER, PK, FK → trips.trip_id)
- `promotion_id` (INTEGER, PK, FK → promotions.promotion_id)
- `discount_applied` (REAL)

### Relationships

```
trips
├── driver_id → drivers.driver_id
├── customer_id → customers.customer_id
├── vehicle_id → vehicles.vehicle_id
├── pickup_city_id → cities.city_id
└── dropoff_city_id → cities.city_id

drivers
└── city_id → cities.city_id

customers
└── city_id → cities.city_id

vehicles
└── driver_id → drivers.driver_id

trip_promotions
├── trip_id → trips.trip_id
└── promotion_id → promotions.promotion_id
```

### Workspaces

The database is organized into 4 workspaces for domain-specific queries:

1. **Mobility** - Trips, drivers, vehicles, cities, customers
2. **Customer Analytics** - Customers, trips, cities
3. **Vehicle Operations** - Vehicles, trips, drivers
4. **Promotions** - Promotions, trip_promotions, trips

## Experiments and Scripts

### 1. Basic QueryGPT Evaluation

Evaluate the QueryGPT system on evaluation data:

```bash
# Evaluate on all queries
python3 evaluate_querygpt.py

# Evaluate on first N queries
python3 evaluate_querygpt.py 10
```

**Output**: `evaluation_results_detailed.json` with metrics for each agent.

### 2. Agent Comparison (Prompt Caching)

Compare standard vs optimized prompt structures for caching:

```bash
python3 compare_agents.py
```

**What it does**:
- Tests standard implementation (mixed static/variable content)
- Tests optimized implementation (static content first, variable last)
- Compares latency and token usage
- Shows cache hit benefits

**Output**: Console output with performance comparison.

### 3. Provider Comparison

Compare different LLM providers (Fireworks vs OpenAI):

```bash
python3 compare_providers.py
```

**Output**: Comparison metrics for accuracy, latency, and cost.

### 4. Fine-Tuning Workflow (SFT)

Complete supervised fine-tuning workflow:

```bash
# Full workflow (baseline → fine-tune → post-tuning evaluation)
python3 run_sft_workflow.py
```

**What it does**:
1. Runs baseline evaluation
2. Uploads dataset to Fireworks
3. Creates fine-tuning job
4. Monitors job progress
5. Deploys fine-tuned model
6. Runs post-tuning evaluation
7. Generates comparison report

**Output**: 
- `evaluation_results_baseline.json`
- `evaluation_results_post_tuning.json`
- `sft_comparison_report.json`

**Prerequisites**:
- Set `FIREWORKS_ACCOUNT_ID` in `.env`
- Dataset file: `golden_dataset_sft.jsonl`

### 5. Column Prune Fine-Tuning

Fine-tune a model specifically for column pruning:

```bash
# Generate dataset for column pruning
python3 generate_column_prune_dataset.py

# Fine-tune the model
python3 fine_tune_column_prune.py

# Evaluate the fine-tuned model
python3 evaluate_column_prune_finetuned.py
```

**Output**: Fine-tuned model for column pruning with improved accuracy.

### 6. Dataset Generation

Generate training datasets:

```bash
# Generate golden dataset for SQL generation
python3 generate_golden_dataset.py

# Generate dataset for fine-tuning
python3 generate_golden_dataset_for_finetune.py

# Generate evaluation data
python3 generate_eval_data.py
```

### 7. Interactive CLI

Interactive command-line interface for testing queries:

```bash
python3 interactive_cli.py
```

**Features**:
- Test individual queries
- See agent outputs at each step
- Debug agent behavior

### 8. Fireworks Features Demo

Demonstrate Fireworks AI features:

```bash
# Run all demos
python3 demo_fireworks_features.py

# Individual demos
python3 demo_prompt_caching.py
python3 demo_pii_protection.py
```

### 9. Example Usage

Simple example of using QueryGPT:

```bash
python3 example_querygpt_usage.py
```

### 10. Debug Agents

Debug individual agent behavior:

```bash
python3 debug_agents.py
```

## Architecture

### 4-Agent Pipeline

1. **Intent Agent** (`llama-v3-8b-instruct`)
   - Classifies question into workspace (mobility, customer_analytics, etc.)
   - Returns: `workspace_id`, `confidence`, `reasoning`

2. **Table Agent** (`firefunction-v2`)
   - Identifies relevant tables for the query
   - Returns: `relevant_tables`, `reasoning`

3. **Column Prune Agent** (`firefunction-v2` or fine-tuned model)
   - Prunes irrelevant columns (40-60% token reduction)
   - Returns: `relevant_columns`, `pruned_schema`, `token_savings_pct`

4. **SQL Generation Agent** (`firefunction-v2`)
   - Generates SQL using few-shot learning
   - Returns: `sql_query`, `explanation`, `tables_used`

### Key Files

- `fireworks_querygpt.py` - Main QueryGPT implementation
- `evaluate_querygpt.py` - Evaluation framework
- `run_sft_workflow.py` - Complete fine-tuning workflow
- `compare_agents.py` - Prompt caching comparison
- `querygpt_workspaces/` - Workspace definitions and schemas
- `evaluation_data.json` - Test queries with ground truth

## Evaluation Metrics

- **Intent Accuracy**: Workspace classification correctness
- **Table Precision/Recall**: Table selection accuracy
- **Column Prune F1**: Column selection accuracy
- **SQL Similarity**: Token-based SQL similarity (0-1)
- **SQL Exact Match**: Exact SQL string match
- **Result Match**: Query result correctness
- **Latency**: Per-agent and total latency (ms)
- **Token Usage**: Input/output tokens per agent
- **Cost**: Estimated cost per query

## Results

Evaluation results are saved in JSON format:

- `evaluation_results_detailed.json` - Detailed per-query results
- `evaluation_results_baseline.json` - Baseline metrics
- `evaluation_results_post_tuning.json` - Post-tuning metrics
- `sft_comparison_report.json` - Before/after comparison

## Common Issues

### Model Not Found Error

If you see "Model not found" errors:
1. Check available models at https://app.fireworks.ai/models
2. Deploy the model if needed
3. Update `FIREWORKS_MODEL` in `.env`

### Tool Calling Not Working

If agents fall back to parsing JSON from content:
- Check model supports function calling
- Verify `tool_choice` parameter is set correctly
- See `TOOL_CALLING_EXPLAINED.md` for details

### Fine-Tuning Job Fails

- Ensure `FIREWORKS_ACCOUNT_ID` is set
- Check dataset format (JSONL with `messages` array)
- Verify API key has fine-tuning permissions

## Documentation

- `AGENT_BREAKDOWN.md` - Detailed agent architecture
- `QUERYGPT_README.md` - QueryGPT implementation guide
- `SFT_WORKFLOW_README.md` - Fine-tuning workflow guide
- `TOOL_CALLING_EXPLAINED.md` - Function calling details
- `FIREWORKS_OPTIMIZATIONS.md` - Optimization techniques
- `DEMO_SLIDES.md` - Presentation slides

## Resources

- [FireworksAI Model Library](https://app.fireworks.ai/models) - Browse available models
- [FireworksAI Documentation](https://fireworks.ai/docs) - API documentation
- [FireworksAI OpenAI SDK](https://fireworks.ai/docs/tools-sdks/openai-compatibility#openai-compatibility) - SDK reference

## License

[Add your license here]
