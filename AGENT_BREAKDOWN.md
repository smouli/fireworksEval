# QueryGPT Agent Breakdown

## Overview

When you run `compare_agents.py`, each agent makes a call to the Fireworks LLM using **function calling** (structured outputs). Each agent returns structured JSON data that gets used by the next agent in the pipeline.

## The 4 Agents

### 1. **Intent Agent** 🔍

**What it does:**
- Takes the user's natural language question
- Classifies it into a business domain workspace (mobility, customer_analytics, vehicle_operations, promotions)
- Uses a smaller, faster model (`llama-v3-8b-instruct`) for quick classification

**Fireworks LLM Call:**
```python
response = client.chat.completions.create(
    model="llama-v3-8b-instruct",
    messages=[{"role": "user", "content": prompt}],
    tools=[intent_function],  # Function calling schema
    tool_choice={"type": "function", "function": {"name": "classify_intent"}},
    temperature=0.1
)
```

**What Fireworks Returns:**
```json
{
    "workspace_id": "mobility",
    "confidence": 0.95,
    "reasoning": "Question is about trips, vehicles, and locations - all mobility domain"
}
```

**Agent Output:**
- `workspace_id`: Which workspace to use (e.g., "mobility")
- `confidence`: How confident (0-1)
- `reasoning`: Why this workspace was chosen

---

### 2. **Table Agent** 📊

**What it does:**
- Takes the question + workspace_id
- Identifies which database tables are needed to answer the question
- Uses the main model (`firefunction-v2`) for better reasoning

**Fireworks LLM Call:**
```python
response = client.chat.completions.create(
    model="firefunction-v2",
    messages=[{"role": "user", "content": prompt}],
    tools=[table_function],
    tool_choice={"type": "function", "function": {"name": "select_tables"}},
    temperature=0.2
)
```

**What Fireworks Returns:**
```json
{
    "relevant_tables": ["trips", "vehicles", "cities"],
    "reasoning": "Need trips for completion status, vehicles for Tesla filter, cities for Seattle filter"
}
```

**Agent Output:**
- `relevant_tables`: List of table names (e.g., ["trips", "vehicles"])
- `reasoning`: Why each table is needed

---

### 3. **Column Prune Agent** ✂️

**What it does:**
- Takes the question + table name + full table schema
- Identifies which columns are needed (SELECT, WHERE, JOIN, GROUP BY, ORDER BY)
- Prunes irrelevant columns to reduce token usage (40-60% reduction)
- Called once per table

**Fireworks LLM Call:**
```python
response = client.chat.completions.create(
    model="firefunction-v2",
    messages=[{"role": "user", "content": prompt}],
    tools=[prune_function],
    tool_choice={"type": "function", "function": {"name": "prune_columns"}},
    temperature=0.1
)
```

**What Fireworks Returns:**
```json
{
    "relevant_columns": ["trip_id", "trip_status", "pickup_time", "pickup_city_id", "vehicle_id"],
    "irrelevant_columns": ["customer_id", "dropoff_time", "payment_method", "rating"],
    "reasoning": "Only need status, time, city, and vehicle for this query"
}
```

**Agent Output:**
- `relevant_columns`: Columns to keep
- `pruned_schema`: Reduced schema with only relevant columns
- `token_savings_pct`: Percentage of tokens saved (e.g., 45%)

---

### 4. **SQL Generation Agent** 💻

**What it does:**
- Takes the question + workspace + relevant tables + pruned schemas + SQL examples
- Generates the final SQL query using few-shot learning
- Uses examples from the workspace to guide generation

**Fireworks LLM Call:**
```python
response = client.chat.completions.create(
    model="firefunction-v2",
    messages=[{"role": "user", "content": prompt}],  # Includes few-shot examples
    tools=[sql_function],
    tool_choice={"type": "function", "function": {"name": "generate_sql"}},
    temperature=0.3
)
```

**What Fireworks Returns:**
```json
{
    "sql_query": "SELECT COUNT(*) as trip_count FROM trips t JOIN vehicles v ON t.vehicle_id = v.vehicle_id JOIN cities c ON t.pickup_city_id = c.city_id WHERE v.make = 'Tesla' AND c.city_name = 'Seattle' AND t.trip_status = 'completed' AND date(t.pickup_time) = date('now', '-1 day')",
    "explanation": "Counts completed trips from yesterday for Tesla vehicles in Seattle by joining trips, vehicles, and cities tables",
    "tables_used": ["trips", "vehicles", "cities"]
}
```

**Agent Output:**
- `sql_query`: The complete SQL query (this is what you see!)
- `explanation`: How the query works
- `tables_used`: Tables used in the query

---

## Complete Pipeline Flow

```
User Question: "How many trips were completed by Teslas in Seattle yesterday?"
    ↓
[1] Intent Agent (Fireworks LLM)
    → Returns: {"workspace_id": "mobility", "confidence": 0.95}
    ↓
[2] Table Agent (Fireworks LLM)
    → Returns: {"relevant_tables": ["trips", "vehicles", "cities"]}
    ↓
[3] Column Prune Agent (Fireworks LLM) - called 3 times (once per table)
    → Returns: {"relevant_columns": [...], "pruned_schema": {...}}
    ↓
[4] SQL Generation Agent (Fireworks LLM)
    → Returns: {"sql_query": "SELECT COUNT(*) ...", "explanation": "..."}
    ↓
Final Output: Complete SQL query ready to execute!
```

## Key Points

1. **All agents use Fireworks LLM** - Each agent makes a separate API call
2. **Function calling ensures structured output** - No parsing needed, guaranteed JSON format
3. **Each agent builds on the previous** - Intent → Tables → Columns → SQL
4. **Fireworks returns structured JSON** - Not free-form text, but validated function call results
5. **The SQL query comes from Agent #4** - That's the final output you see

## What You See in compare_agents.py

When you run `compare_agents.py`, you see:
- **Latency**: Total time for all 4 agents combined
- **SQL**: The final SQL query from Agent #4
- **Metrics**: Performance stats for each agent (latency, tokens, success rate)

The comparison shows:
- **Standard**: Mixed prompt structure (less cache-friendly)
- **Optimized**: Static content first, variable last (better cache hits)

Both versions use the same 4 agents - the difference is just how prompts are structured for caching!

