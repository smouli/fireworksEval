#!/usr/bin/env python3
"""
Generate training dataset for Column Prune Agent in JSON mode format.
Extracts relevant columns from SQL queries in evaluation_data.json.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set

def extract_columns_from_sql(sql: str) -> Set[str]:
    """Extract all column names used in SQL query."""
    columns = set()
    
    # Extract SELECT columns
    select_match = re.search(r'SELECT\s+(.+?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
    if select_match:
        cols_str = select_match.group(1)
        # Handle DISTINCT, CASE, etc.
        cols = [c.strip() for c in cols_str.split(',')]
        for col in cols:
            # Remove aliases, functions, table prefixes
            col_name = re.sub(r'\s+as\s+\w+$', '', col, flags=re.IGNORECASE).strip()
            col_name = re.sub(r'^\w+\.', '', col_name).strip()
            col_name = re.sub(r'\(.*\)', '', col_name).strip()  # Remove function calls
            # Extract column name from functions like COUNT(*), SUM(amount)
            if '(' in col_name and ')' in col_name:
                # Try to extract column name from function
                func_match = re.search(r'\((\w+)\)', col_name)
                if func_match:
                    col_name = func_match.group(1)
                else:
                    continue  # Skip COUNT(*), etc.
            if col_name and col_name.upper() not in ['DISTINCT', 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN', '*']:
                columns.add(col_name)
    
    # Extract WHERE columns
    where_match = re.search(r'WHERE\s+(.+?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)
    if where_match:
        where_clause = where_match.group(1)
        # Find column names (simple pattern - column = value or column IN (...))
        col_patterns = [
            r'\b([a-z_][a-z0-9_]*)\s*[=<>!]',  # column = value
            r'\b([a-z_][a-z0-9_]*)\s+IN\s*\(',  # column IN (...)
            r'\b([a-z_][a-z0-9_]*)\s+LIKE',     # column LIKE
            r'\b([a-z_][a-z0-9_]*)\s+IS\s+NULL', # column IS NULL
        ]
        for pattern in col_patterns:
            for match in re.finditer(pattern, where_clause, re.IGNORECASE):
                columns.add(match.group(1))
    
    # Extract JOIN columns
    join_pattern = r'JOIN\s+\w+\s+\w+\s+ON\s+(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)'
    for match in re.finditer(join_pattern, sql, re.IGNORECASE):
        columns.add(match.group(2))
        columns.add(match.group(4))
    
    # Extract GROUP BY columns
    group_match = re.search(r'GROUP\s+BY\s+(.+?)(?:\s+ORDER\s+BY|\s+LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)
    if group_match:
        cols_str = group_match.group(1)
        cols = [c.strip() for c in cols_str.split(',')]
        for col in cols:
            col_name = re.sub(r'^\w+\.', '', col).strip()
            columns.add(col_name)
    
    # Extract ORDER BY columns
    order_match = re.search(r'ORDER\s+BY\s+(.+?)(?:\s+LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)
    if order_match:
        cols_str = order_match.group(1)
        cols = [c.strip() for c in cols_str.split(',')]
        for col in cols:
            col_name = re.sub(r'\s+(ASC|DESC)$', '', col, flags=re.IGNORECASE).strip()
            col_name = re.sub(r'^\w+\.', '', col_name).strip()
            columns.add(col_name)
    
    return columns

def extract_tables_from_sql(sql: str) -> List[str]:
    """Extract table names from SQL query."""
    tables = set()
    # FROM clause
    from_match = re.search(r'FROM\s+(\w+)', sql, re.IGNORECASE)
    if from_match:
        tables.add(from_match.group(1))
    # JOIN clauses
    tables.update(re.findall(r'JOIN\s+(\w+)', sql, re.IGNORECASE))
    return list(tables)

def determine_workspace(question: str, sql: str, workspaces: Dict) -> str:
    """Determine workspace from question or SQL."""
    question_lower = question.lower()
    sql_lower = sql.lower()
    
    # Check keywords in question
    for ws_id, ws in workspaces.items():
        keywords = ws.get("keywords", [])
        if any(keyword in question_lower for keyword in keywords):
            return ws_id
    
    # Check tables in SQL
    tables = extract_tables_from_sql(sql)
    for ws_id, ws in workspaces.items():
        workspace_tables = ws.get("tables", [])
        if any(table in workspace_tables for table in tables):
            return ws_id
    
    # Default to mobility (most common)
    return "mobility"

def generate_column_prune_dataset(
    evaluation_data_path: str = "evaluation_data.json",
    workspaces_path: str = "querygpt_workspaces/workspaces.json",
    output_path: str = "column_prune_dataset.jsonl",
    target_size: int = 500
) -> None:
    """Generate training dataset for column prune agent."""
    
    # Load evaluation data
    print(f"📖 Loading evaluation data from {evaluation_data_path}...")
    with open(evaluation_data_path, 'r') as f:
        eval_data = json.load(f)
    print(f"✅ Loaded {len(eval_data)} evaluation examples")
    
    # Load workspaces to get table schemas
    print(f"📖 Loading workspaces from {workspaces_path}...")
    with open(workspaces_path, 'r') as f:
        workspaces = json.load(f)
    print(f"✅ Loaded {len(workspaces)} workspaces")
    
    training_examples = []
    
    print(f"\n🔍 Processing evaluation data...")
    for idx, test_case in enumerate(eval_data):
        question = test_case["question"]
        sql = test_case["sql"]
        
        # Extract columns from SQL
        relevant_cols = extract_columns_from_sql(sql)
        
        if not relevant_cols:
            continue
        
        # Determine workspace
        workspace_id = determine_workspace(question, sql, workspaces)
        workspace = workspaces[workspace_id]
        
        # Find which table(s) are used
        tables_in_sql = extract_tables_from_sql(sql)
        if not tables_in_sql:
            continue
        
        # Generate training example for each table
        for table_name in tables_in_sql:
            if table_name not in workspace.get("table_schemas", {}):
                continue
            
            full_schema = workspace["table_schemas"][table_name]
            all_columns = [col["name"] for col in full_schema["columns"]]
            
            # Filter relevant columns to only those in this table
            table_relevant = [c for c in relevant_cols if c in all_columns]
            table_irrelevant = [c for c in all_columns if c not in table_relevant]
            
            # Always include primary keys
            primary_keys = [col["name"] for col in full_schema["columns"] if col.get("primary_key")]
            for pk in primary_keys:
                if pk not in table_relevant:
                    table_relevant.append(pk)
                    if pk in table_irrelevant:
                        table_irrelevant.remove(pk)
            
            if not table_relevant:
                continue
            
            # Build prompt (same format as agent uses)
            columns_info = "\n".join([
                f"- {col['name']} ({col['type']})" + (" [PK]" if col.get("primary_key") else "")
                for col in full_schema["columns"]
            ])
            
            prompt = f"""You are a column pruning agent. Analyze which columns are needed for the query.

Table: {table_name}
Question: {question}

Available columns:
{columns_info}

Identify:
1. Columns needed in SELECT clause
2. Columns needed in WHERE/JOIN conditions
3. Columns needed for GROUP BY/ORDER BY
4. Primary keys and foreign keys (always keep these)

Prune all other columns to reduce token usage."""
            
            # Build JSON response
            response_json = json.dumps({
                "relevant_columns": table_relevant,
                "irrelevant_columns": table_irrelevant,
                "reasoning": f"Columns needed for query: {', '.join(table_relevant)}"
            }, ensure_ascii=False)
            
            training_examples.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response_json}
                ]
            })
        
        if (idx + 1) % 10 == 0:
            print(f"   Processed {idx + 1}/{len(eval_data)} examples, generated {len(training_examples)} training examples")
    
    print(f"\n✅ Generated {len(training_examples)} training examples")
    
    # Expand dataset to target size if needed
    if len(training_examples) < target_size:
        print(f"📈 Expanding dataset from {len(training_examples)} to {target_size} examples...")
        original_examples = training_examples.copy()
        while len(training_examples) < target_size:
            for example in original_examples:
                if len(training_examples) >= target_size:
                    break
                training_examples.append(example)
        print(f"✅ Expanded to {len(training_examples)} examples")
    
    # Write JSONL file
    print(f"\n💾 Writing dataset to {output_path}...")
    with open(output_path, 'w') as f:
        for example in training_examples[:target_size]:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved {len(training_examples[:target_size])} training examples to {output_path}")

if __name__ == "__main__":
    import sys
    target_size = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    generate_column_prune_dataset(target_size=target_size)

