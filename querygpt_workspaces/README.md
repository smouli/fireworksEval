# QueryGPT-Style Workspace System

This directory contains a QueryGPT-inspired workspace system for organizing SQL samples, schemas, and metadata for text-to-SQL generation.

## Overview

The system mimics Uber's QueryGPT architecture with:
- **Workspaces**: Curated collections of SQL samples and tables for specific business domains
- **Intent Agent**: Maps user questions to appropriate workspaces
- **Table Agent**: Identifies relevant tables for queries
- **Column Prune Agent**: Reduces schema size by pruning irrelevant columns

## Structure

### `workspaces.json`
Contains 4 system workspaces:
- **Mobility**: Trips, drivers, vehicles, cities (8 SQL samples)
- **Customer Analytics**: Customer behavior and spending (4 SQL samples)
- **Vehicle Operations**: Vehicle usage and performance (3 SQL samples)
- **Promotions**: Promotions and discounts (3 SQL samples)

Each workspace includes:
- Table schemas with column details
- SQL samples with questions and queries
- Keywords for intent matching
- Table mappings

### `intent_mapping.json`
Examples and categories for the Intent Agent:
- Intent examples with reasoning
- Intent categories with keywords and primary tables

### `table_agent.json`
Examples for the Table Agent:
- Table selection examples
- Reasoning for table relevance

### `column_prune.json`
Examples for the Column Prune Agent:
- Column pruning examples
- Strategy for identifying relevant columns

## Usage Example

```python
import json

# Load workspaces
with open('querygpt_workspaces/workspaces.json') as f:
    workspaces = json.load(f)

# Get Mobility workspace
mobility = workspaces['mobility']

# Access SQL samples for few-shot learning
for sample in mobility['sql_samples']:
    print(f"Q: {sample['question']}")
    print(f"SQL: {sample['sql']}\n")

# Access table schemas
trips_schema = mobility['table_schemas']['trips']
print(f"Trips table has {len(trips_schema['columns'])} columns")
```

## QueryGPT Architecture Flow

1. **User Question** → "How many trips were completed by Teslas in Seattle yesterday?"

2. **Intent Agent** → Maps to "mobility" workspace
   - Uses keywords: "trips", "Tesla", "Seattle"
   - Returns workspace: "mobility"

3. **Table Agent** → Identifies relevant tables
   - Input: Question + workspace tables
   - Output: ["trips", "vehicles", "cities"]
   - User can confirm or edit

4. **Column Prune Agent** → Reduces schema size
   - Input: Question + selected tables
   - Output: Pruned schemas with only relevant columns
   - Example: trips table → only [trip_id, trip_status, pickup_time, pickup_city_id, vehicle_id]

5. **RAG + LLM** → Generate SQL
   - Few-shot examples from workspace SQL samples
   - Pruned table schemas
   - User question
   - → Generated SQL query

## Workspace Details

### Mobility Workspace
- **Tables**: trips, drivers, vehicles, cities, customers
- **Keywords**: trip, driver, vehicle, ride, pickup, dropoff, city, miles, duration, fare, rating
- **Use Cases**: Trip analysis, driver performance, city-based queries, vehicle usage

### Customer Analytics Workspace
- **Tables**: customers, trips, cities
- **Keywords**: customer, spending, lifetime, value, signup, revenue
- **Use Cases**: Customer segmentation, spending analysis, signup trends

### Vehicle Operations Workspace
- **Tables**: vehicles, trips, drivers
- **Keywords**: vehicle, car, make, model, type, license
- **Use Cases**: Vehicle performance, usage statistics, fleet management

### Promotions Workspace
- **Tables**: promotions, trip_promotions, trips
- **Keywords**: promotion, discount, code, campaign
- **Use Cases**: Promotion effectiveness, discount tracking, campaign analysis

## Adding New Workspaces

To add a custom workspace:

1. Add workspace definition to `create_querygpt_workspaces.py`
2. Include:
   - Name and description
   - Relevant tables
   - Keywords for intent matching
   - SQL samples (5-10 examples)
3. Run `python3 create_querygpt_workspaces.py` to regenerate

## Notes

- SQL samples are validated against the actual database
- Table schemas include foreign key relationships
- Column pruning examples show token reduction strategies
- Intent mapping helps narrow RAG search radius

