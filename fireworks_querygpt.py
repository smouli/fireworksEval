#!/usr/bin/env python3
"""
QueryGPT Agents using Fireworks AI with function calling for structured outputs.
Leverages Fireworks features for fast inference and accurate tool calling.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI
from dataclasses import dataclass
from pathlib import Path


def load_env_file(env_path: str = ".env") -> None:
    """Load environment variables from .env file."""
    env_file = Path(env_path)
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    # Remove quotes if present
                    value = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = value


# Load .env file if it exists
load_env_file()


def validate_intent_result(result: Dict[str, Any], valid_workspaces: List[str]) -> Tuple[bool, List[str]]:
    """Validate intent agent result against schema."""
    errors = []
    
    # Check required fields
    required_fields = ["workspace_id", "confidence", "reasoning"]
    for field in required_fields:
        if field not in result:
            errors.append(f"Missing required field: {field}")
    
    # Validate workspace_id enum
    if "workspace_id" in result:
        if result["workspace_id"] not in valid_workspaces:
            errors.append(f"Invalid workspace_id: {result['workspace_id']}. Must be one of {valid_workspaces}")
    
    # Validate confidence is a number between 0 and 1
    if "confidence" in result:
        try:
            conf = float(result["confidence"])
            if conf < 0 or conf > 1:
                errors.append(f"Confidence out of range: {conf}. Must be between 0 and 1")
        except (ValueError, TypeError):
            errors.append(f"Confidence is not a valid number: {result['confidence']}")
    
    # Validate reasoning is a string
    if "reasoning" in result and not isinstance(result["reasoning"], str):
        errors.append(f"Reasoning must be a string, got {type(result['reasoning'])}")
    
    return len(errors) == 0, errors


def validate_table_result(result: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate table agent result against schema."""
    errors = []
    
    # Check required fields
    if "relevant_tables" not in result:
        errors.append("Missing required field: relevant_tables")
    elif not isinstance(result["relevant_tables"], list):
        errors.append(f"relevant_tables must be a list, got {type(result['relevant_tables'])}")
    
    if "reasoning" not in result:
        errors.append("Missing required field: reasoning")
    elif not isinstance(result["reasoning"], str):
        errors.append(f"reasoning must be a string, got {type(result['reasoning'])}")
    
    return len(errors) == 0, errors


def validate_prune_result(result: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate column prune agent result against schema."""
    errors = []
    
    # Check required fields
    required_fields = ["relevant_columns", "irrelevant_columns", "reasoning"]
    for field in required_fields:
        if field not in result:
            errors.append(f"Missing required field: {field}")
    
    # Validate columns are lists
    if "relevant_columns" in result and not isinstance(result["relevant_columns"], list):
        errors.append(f"relevant_columns must be a list, got {type(result['relevant_columns'])}")
    
    if "irrelevant_columns" in result and not isinstance(result["irrelevant_columns"], list):
        errors.append(f"irrelevant_columns must be a list, got {type(result['irrelevant_columns'])}")
    
    if "reasoning" in result and not isinstance(result["reasoning"], str):
        errors.append(f"reasoning must be a string, got {type(result['reasoning'])}")
    
    return len(errors) == 0, errors


def validate_sql_result(result: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate SQL generation agent result against schema."""
    errors = []
    
    # Check required fields
    required_fields = ["sql_query", "explanation", "tables_used"]
    for field in required_fields:
        if field not in result:
            errors.append(f"Missing required field: {field}")
    
    # Validate types
    if "sql_query" in result and not isinstance(result["sql_query"], str):
        errors.append(f"sql_query must be a string, got {type(result['sql_query'])}")
    
    if "explanation" in result and not isinstance(result["explanation"], str):
        errors.append(f"explanation must be a string, got {type(result['explanation'])}")
    
    if "tables_used" in result:
        if not isinstance(result["tables_used"], list):
            errors.append(f"tables_used must be a list, got {type(result['tables_used'])}")
        elif not all(isinstance(t, str) for t in result["tables_used"]):
            errors.append("All items in tables_used must be strings")
    
    return len(errors) == 0, errors


@dataclass
class AgentMetrics:
    """Track metrics for each agent call."""
    agent_name: str
    latency_ms: float
    tokens_used: int
    function_called: bool
    accuracy: Optional[float] = None
    error: Optional[str] = None
    used_tool_calls: Optional[bool] = None  # True if structured tool_calls used, False if fallback
    tool_call_accuracy: Optional[bool] = None  # True if tool call was valid (schema, enums, types)
    validation_errors: Optional[List[str]] = None  # List of validation errors if any


class FireworksQueryGPT:
    """QueryGPT implementation using Fireworks AI with function calling."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.fireworks.ai/inference/v1",
        model: Optional[str] = None,  # Will use environment variable or default
        intent_model: Optional[str] = None,  # Will use environment variable or default
        column_prune_model: Optional[str] = None,  # Optional: fine-tuned model for column pruning (JSON mode)
    ):
        self.client = OpenAI(
            api_key=api_key or os.getenv("FIREWORKS_API_KEY"),
            base_url=base_url
        )
        # Use environment variables or defaults - users should check https://app.fireworks.ai/models
        self.model = model or os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/firefunction-v2")
        self.intent_model = intent_model or os.getenv("FIREWORKS_INTENT_MODEL", "accounts/fireworks/models/llama-v3-8b-instruct")
        self.column_prune_model = column_prune_model  # Fine-tuned model for column pruning (uses JSON mode)
        self.metrics: List[AgentMetrics] = []
        
        # Load workspace data
        self.workspaces = self._load_workspaces()
        self.intent_mapping = self._load_intent_mapping()
    
    def _load_workspaces(self) -> Dict:
        with open("querygpt_workspaces/workspaces.json") as f:
            return json.load(f)
    
    def _load_intent_mapping(self) -> Dict:
        with open("querygpt_workspaces/intent_mapping.json") as f:
            return json.load(f)
    
    def intent_agent(self, question: str) -> Dict[str, Any]:
        """
        Intent Agent: Maps user question to workspace using function calling.
        Uses smaller model for fast classification.
        
        Fireworks API Features Used:
        - Function calling (structured outputs) - API feature
        - Fast inference models - API feature (llama-v3p3-8b-instruct is optimized)
        - Low temperature for consistency - API parameter
        """
        start_time = time.time()
        
        # Function schema for structured output (Fireworks API feature)
        intent_function = {
            "type": "function",
            "function": {
                "name": "classify_intent",
                "description": "Classify the user's question into a business domain workspace",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workspace_id": {
                            "type": "string",
                            "enum": list(self.workspaces.keys()),
                            "description": "The workspace ID that best matches the question"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence score between 0 and 1"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief explanation of why this workspace was chosen"
                        }
                    },
                    "required": ["workspace_id", "confidence", "reasoning"]
                }
            }
        }
        
        # Build prompt with workspace descriptions
        workspace_descriptions = "\n".join([
            f"- {ws_id}: {ws['name']} - {ws['description']}"
            for ws_id, ws in self.workspaces.items()
        ])
        
        prompt = f"""You are an intent classification agent for a text-to-SQL system.

Available workspaces:
{workspace_descriptions}

User question: {question}

Classify this question into the most appropriate workspace. Consider keywords, domain, and query intent."""

        try:
            response = self.client.chat.completions.create(
                model=self.intent_model,
                messages=[{"role": "user", "content": prompt}],
                tools=[intent_function],
                tool_choice={"type": "function", "function": {"name": "classify_intent"}},
                temperature=0.1,  # Low temperature for consistent classification
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Check if tool_calls exists, otherwise try to parse from content
            message = response.choices[0].message
            used_tool_calls = False
            validation_errors = []
            
            # Debug: Print what we actually received
            print(f"  🔍 Debug - message.tool_calls: {message.tool_calls}")
            print(f"  🔍 Debug - message.content: {message.content[:200] if message.content else None}")
            
            if message.tool_calls:
                print(f"  ✅ Intent Agent: Using structured tool_calls")
                function_call = message.tool_calls[0]
                # Validate function name
                if function_call.function.name != "classify_intent":
                    validation_errors.append(f"Wrong function name: {function_call.function.name}")
                
                try:
                    result = json.loads(function_call.function.arguments)
                    used_tool_calls = True
                except json.JSONDecodeError as e:
                    validation_errors.append(f"Failed to parse function arguments as JSON: {e}")
                    raise
            else:
                # Fallback: parse function call from content if model returns it as text
                print(f"  ⚠️  Intent Agent: Fallback - Parsing from content (not true tool calling)")
                print(f"  ⚠️  This means the model is NOT using structured tool_calls API format")
                content = message.content or ""
                try:
                    # Try to parse JSON from content
                    if content.strip().startswith("{"):
                        func_data = json.loads(content)
                        if "parameters" in func_data:
                            result = func_data["parameters"]
                        elif "name" in func_data and func_data["name"] == "classify_intent":
                            # Handle case where model returns function call as JSON
                            result = func_data.get("parameters", func_data)
                        else:
                            result = func_data
                    else:
                        raise ValueError(f"Model did not return function call. Response: {content}")
                except json.JSONDecodeError:
                    raise ValueError(f"Model did not return valid function call. Response: {content}")
            
            # Validate the result against schema
            is_valid, val_errors = validate_intent_result(result, list(self.workspaces.keys()))
            validation_errors.extend(val_errors)
            
            if validation_errors:
                print(f"  ⚠️  Validation errors: {validation_errors}")
            
            self.metrics.append(AgentMetrics(
                agent_name="intent_agent",
                latency_ms=latency_ms,
                tokens_used=response.usage.total_tokens,
                function_called=True,
                accuracy=result.get("confidence"),
                used_tool_calls=used_tool_calls,
                tool_call_accuracy=is_valid,
                validation_errors=validation_errors if validation_errors else None
            ))
            
            return result
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.append(AgentMetrics(
                agent_name="intent_agent",
                latency_ms=latency_ms,
                tokens_used=0,
                function_called=False,
                error=str(e)
            ))
            raise
    
    def table_agent(self, question: str, workspace_id: str) -> Dict[str, Any]:
        """
        Table Agent: Identifies relevant tables using function calling.
        Uses main model for better reasoning.
        
        Fireworks API Features Used:
        - Function calling for structured outputs - API feature
        - Model selection (larger model for reasoning) - API feature
        """
        start_time = time.time()
        workspace = self.workspaces[workspace_id]
        available_tables = workspace["tables"]
        
        # Function schema for table selection (Fireworks API feature)
        table_function = {
            "type": "function",
            "function": {
                "name": "select_tables",
                "description": "Select the relevant tables needed to answer the question",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "relevant_tables": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of table names needed for the query"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Explanation of why each table is needed"
                        }
                    },
                    "required": ["relevant_tables", "reasoning"]
                }
            }
        }
        
        prompt = f"""You are a table selection agent. Given a user question and available tables in the workspace, identify which tables are needed.

Workspace: {workspace['name']}
Available tables: {', '.join(available_tables)}

User question: {question}

Select only the tables that are necessary to answer this question. Consider:
- Tables mentioned in the question
- Tables needed for joins
- Tables containing required data fields"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=[table_function],
                tool_choice={"type": "function", "function": {"name": "select_tables"}},
                temperature=0.2,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Check if tool_calls exists, otherwise try to parse from content
            message = response.choices[0].message
            used_tool_calls = False
            validation_errors = []
            
            if message.tool_calls:
                print(f"  ✅ Table Agent: Using structured tool_calls")
                function_call = message.tool_calls[0]
                if function_call.function.name != "select_tables":
                    validation_errors.append(f"Wrong function name: {function_call.function.name}")
                try:
                    result = json.loads(function_call.function.arguments)
                    used_tool_calls = True
                except json.JSONDecodeError as e:
                    validation_errors.append(f"Failed to parse function arguments as JSON: {e}")
                    raise
            else:
                # Fallback: parse function call from content if model returns it as text
                print(f"  ⚠️  Table Agent: Fallback - Parsing from content (not true tool calling)")
                content = message.content or ""
                try:
                    if content.strip().startswith("{"):
                        func_data = json.loads(content)
                        if "parameters" in func_data:
                            result = func_data["parameters"]
                        elif "name" in func_data and func_data["name"] == "select_tables":
                            result = func_data.get("parameters", func_data)
                        else:
                            result = func_data
                    else:
                        raise ValueError(f"Model did not return function call. Response: {content}")
                except json.JSONDecodeError:
                    raise ValueError(f"Model did not return valid function call. Response: {content}")
            
            # Validate the result against schema
            is_valid, val_errors = validate_table_result(result)
            validation_errors.extend(val_errors)
            
            # Validate tables exist in workspace
            if "relevant_tables" in result:
                valid_tables = [t for t in result["relevant_tables"] if t in available_tables]
                invalid_tables = [t for t in result["relevant_tables"] if t not in available_tables]
                if invalid_tables:
                    validation_errors.append(f"Invalid tables selected: {invalid_tables}. Available: {available_tables}")
            else:
                valid_tables = []
            
            if validation_errors:
                print(f"  ⚠️  Validation errors: {validation_errors}")
            
            self.metrics.append(AgentMetrics(
                agent_name="table_agent",
                latency_ms=latency_ms,
                tokens_used=response.usage.total_tokens,
                function_called=True,
                used_tool_calls=used_tool_calls,
                tool_call_accuracy=is_valid and len(validation_errors) == 0,
                validation_errors=validation_errors if validation_errors else None
            ))
            
            return {
                "relevant_tables": valid_tables,
                "reasoning": result["reasoning"]
            }
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.append(AgentMetrics(
                agent_name="table_agent",
                latency_ms=latency_ms,
                tokens_used=0,
                function_called=False,
                error=str(e)
            ))
            raise
    
    def column_prune_agent(self, question: str, table_name: str, full_schema: Dict, use_json_mode: bool = False, json_model: Optional[str] = None) -> Dict[str, Any]:
        """
        Column Prune Agent: Identifies relevant columns.
        Can use JSON mode (for fine-tuned models) or tool calling (default).
        Reduces token usage significantly.
        
        Fireworks API Features Used:
        - Function calling for structured outputs - API feature (default)
        - JSON mode for fine-tuned models (when use_json_mode=True)
        
        Code Optimization:
        - Column pruning logic to reduce input tokens (our code)
        """
        start_time = time.time()
        all_columns = [col["name"] for col in full_schema["columns"]]
        
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

        try:
            if use_json_mode and json_model:
                # JSON mode for fine-tuned model
                print(f"  📝 Column Prune Agent: Using JSON mode with model {json_model}")
                response = self.client.chat.completions.create(
                    model=json_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
                
                latency_ms = (time.time() - start_time) * 1000
                content = response.choices[0].message.content
                try:
                    result = json.loads(content)
                    used_tool_calls = False  # JSON mode, not tool calling
                    validation_errors = []
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse JSON response: {e}. Content: {content}")
            else:
                # Original tool calling mode
                prune_function = {
                    "type": "function",
                    "function": {
                        "name": "prune_columns",
                        "description": "Identify which columns are needed for the query",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "relevant_columns": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Column names needed for the query"
                                },
                                "irrelevant_columns": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Column names that can be pruned"
                                },
                                "reasoning": {
                                    "type": "string",
                                    "description": "Explanation of column selection"
                                }
                            },
                            "required": ["relevant_columns", "irrelevant_columns", "reasoning"]
                        }
                    }
                }
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    tools=[prune_function],
                    tool_choice={"type": "function", "function": {"name": "prune_columns"}},
                    temperature=0.1,
                )
                
                latency_ms = (time.time() - start_time) * 1000
                
                # Check if tool_calls exists, otherwise try to parse from content
                message = response.choices[0].message
                used_tool_calls = False
                validation_errors = []
                
                if message.tool_calls:
                    print(f"  ✅ Column Prune Agent: Using structured tool_calls")
                    function_call = message.tool_calls[0]
                    if function_call.function.name != "prune_columns":
                        validation_errors.append(f"Wrong function name: {function_call.function.name}")
                    try:
                        result = json.loads(function_call.function.arguments)
                        used_tool_calls = True
                    except json.JSONDecodeError as e:
                        validation_errors.append(f"Failed to parse function arguments as JSON: {e}")
                        raise
                else:
                    # Fallback: parse function call from content if model returns it as text
                    print(f"  ⚠️  Column Prune Agent: Fallback - Parsing from content (not true tool calling)")
                    content = message.content or ""
                    try:
                        if content.strip().startswith("{"):
                            func_data = json.loads(content)
                            if "parameters" in func_data:
                                result = func_data["parameters"]
                            elif "name" in func_data and func_data["name"] == "prune_columns":
                                result = func_data.get("parameters", func_data)
                            else:
                                result = func_data
                        else:
                            raise ValueError(f"Model did not return function call. Response: {content}")
                    except json.JSONDecodeError:
                        raise ValueError(f"Model did not return valid function call. Response: {content}")
            
            # Validate the result against schema
            is_valid, val_errors = validate_prune_result(result)
            validation_errors.extend(val_errors)
            
            # Validate columns exist
            if "relevant_columns" in result:
                valid_relevant = [c for c in result["relevant_columns"] if c in all_columns]
                invalid_relevant = [c for c in result["relevant_columns"] if c not in all_columns]
                if invalid_relevant:
                    validation_errors.append(f"Invalid relevant_columns: {invalid_relevant}")
            else:
                valid_relevant = []
            
            if "irrelevant_columns" in result:
                valid_irrelevant = [c for c in result["irrelevant_columns"] if c in all_columns]
                invalid_irrelevant = [c for c in result["irrelevant_columns"] if c not in all_columns]
                if invalid_irrelevant:
                    validation_errors.append(f"Invalid irrelevant_columns: {invalid_irrelevant}")
            else:
                valid_irrelevant = []
            
            # Calculate token savings (our code optimization)
            original_tokens = len(str(full_schema))
            pruned_schema = {
                "table_name": table_name,
                "columns": [col for col in full_schema["columns"] if col["name"] in valid_relevant]
            }
            pruned_tokens = len(str(pruned_schema))
            savings_pct = ((original_tokens - pruned_tokens) / original_tokens) * 100 if original_tokens > 0 else 0
            
            if validation_errors:
                print(f"  ⚠️  Validation errors: {validation_errors}")
            
            self.metrics.append(AgentMetrics(
                agent_name="column_prune_agent",
                latency_ms=latency_ms,
                tokens_used=response.usage.total_tokens,
                function_called=True,
                accuracy=savings_pct,
                used_tool_calls=used_tool_calls,
                tool_call_accuracy=is_valid and len(validation_errors) == 0,
                validation_errors=validation_errors if validation_errors else None
            ))
            
            return {
                "relevant_columns": valid_relevant,
                "pruned_schema": pruned_schema,
                "token_savings_pct": savings_pct
            }
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.append(AgentMetrics(
                agent_name="column_prune_agent",
                latency_ms=latency_ms,
                tokens_used=0,
                function_called=False,
                error=str(e)
            ))
            raise
    
    def sql_generation_agent(
        self,
        question: str,
        workspace_id: str,
        relevant_tables: List[str],
        pruned_schemas: Dict[str, Dict],
        sql_samples: List[Dict]
    ) -> Dict[str, Any]:
        """
        SQL Generation Agent: Generates SQL using few-shot learning.
        Uses function calling for structured SQL output.
        
        Fireworks API Features Used:
        - Function calling for structured outputs - API feature
        - Few-shot learning via prompts - API feature (we structure the prompt)
        - Temperature control - API parameter
        """
        start_time = time.time()
        workspace = self.workspaces[workspace_id]
        
        # Function schema for SQL generation (Fireworks API feature)
        sql_function = {
            "type": "function",
            "function": {
                "name": "generate_sql",
                "description": "Generate a SQL query to answer the user's question",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql_query": {
                            "type": "string",
                            "description": "The complete SQL query"
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Brief explanation of how the query works"
                        },
                        "tables_used": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of tables used in the query"
                        }
                    },
                    "required": ["sql_query", "explanation", "tables_used"]
                }
            }
        }
        
        # Build few-shot examples (our code - structuring prompts)
        few_shot_examples = "\n\n".join([
            f"Q: {sample['question']}\nSQL: {sample['sql']}"
            for sample in sql_samples[:5]  # Top 5 most relevant
        ])
        
        # Build schema context (pruned - our code optimization)
        schema_context = "\n\n".join([
            f"Table: {table_name}\nColumns: {', '.join([col['name'] for col in schema['columns']])}"
            for table_name, schema in pruned_schemas.items()
        ])
        
        prompt = f"""You are a SQL generation agent. Generate a SQLite query to answer the user's question.

Workspace: {workspace['name']}
Domain: {workspace['description']}

Few-shot examples:
{few_shot_examples}

Table schemas (pruned):
{schema_context}

User question: {question}

Generate a valid SQLite query. Ensure:
- Correct table and column names
- Proper JOIN syntax
- Valid WHERE/GROUP BY/ORDER BY clauses
- SQLite-compatible functions"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=[sql_function],
                tool_choice={"type": "function", "function": {"name": "generate_sql"}},
                temperature=0.3,  # Some creativity but still structured
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Check if tool_calls exists, otherwise try to parse from content
            message = response.choices[0].message
            used_tool_calls = False
            validation_errors = []
            
            if message.tool_calls:
                print(f"  ✅ SQL Generation Agent: Using structured tool_calls")
                function_call = message.tool_calls[0]
                if function_call.function.name != "generate_sql":
                    validation_errors.append(f"Wrong function name: {function_call.function.name}")
                try:
                    result = json.loads(function_call.function.arguments)
                    used_tool_calls = True
                except json.JSONDecodeError as e:
                    validation_errors.append(f"Failed to parse function arguments as JSON: {e}")
                    raise
            else:
                # Fallback: parse function call from content if model returns it as text
                print(f"  ⚠️  SQL Generation Agent: Fallback - Parsing from content (not true tool calling)")
                content = message.content or ""
                try:
                    if content.strip().startswith("{"):
                        func_data = json.loads(content)
                        if "parameters" in func_data:
                            result = func_data["parameters"]
                        elif "name" in func_data and func_data["name"] == "generate_sql":
                            result = func_data.get("parameters", func_data)
                        else:
                            result = func_data
                    else:
                        raise ValueError(f"Model did not return function call. Response: {content}")
                except json.JSONDecodeError:
                    raise ValueError(f"Model did not return valid function call. Response: {content}")
            
            # Validate the result against schema
            is_valid, val_errors = validate_sql_result(result)
            validation_errors.extend(val_errors)
            
            if validation_errors:
                print(f"  ⚠️  Validation errors: {validation_errors}")
            
            self.metrics.append(AgentMetrics(
                agent_name="sql_generation_agent",
                latency_ms=latency_ms,
                tokens_used=response.usage.total_tokens,
                function_called=True,
                used_tool_calls=used_tool_calls,
                tool_call_accuracy=is_valid and len(validation_errors) == 0,
                validation_errors=validation_errors if validation_errors else None
            ))
            
            return result
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.append(AgentMetrics(
                agent_name="sql_generation_agent",
                latency_ms=latency_ms,
                tokens_used=0,
                function_called=False,
                error=str(e)
            ))
            raise
    
    def generate_sql(self, question: str) -> Dict[str, Any]:
        """
        Complete QueryGPT pipeline: Intent → Table → Column Prune → SQL Generation
        
        Code Optimization:
        - Sequential pipeline (could be parallelized with asyncio - our code)
        """
        pipeline_start = time.time()
        
        try:
            # Step 1: Intent Agent
            intent_result = self.intent_agent(question)
            workspace_id = intent_result["workspace_id"]
            workspace = self.workspaces[workspace_id]
            
            # Step 2: Table Agent
            table_result = self.table_agent(question, workspace_id)
            relevant_tables = table_result["relevant_tables"]
            
            # Step 3: Column Prune Agent (for each table)
            # Could parallelize this with asyncio (our code optimization)
            pruned_schemas = {}
            for table_name in relevant_tables:
                full_schema = workspace["table_schemas"][table_name]
                # Use JSON mode if fine-tuned model is specified
                use_json_mode = self.column_prune_model is not None
                prune_result = self.column_prune_agent(
                    question, 
                    table_name, 
                    full_schema,
                    use_json_mode=use_json_mode,
                    json_model=self.column_prune_model
                )
                pruned_schemas[table_name] = prune_result["pruned_schema"]
            
            # Step 4: Get relevant SQL samples
            sql_samples = workspace["sql_samples"]
            
            # Step 5: SQL Generation
            sql_result = self.sql_generation_agent(
                question,
                workspace_id,
                relevant_tables,
                pruned_schemas,
                sql_samples
            )
            
            total_latency = (time.time() - pipeline_start) * 1000
            
            return {
                "question": question,
                "workspace": workspace_id,
                "tables": relevant_tables,
                "sql": sql_result["sql_query"],
                "explanation": sql_result["explanation"],
                "intent_confidence": intent_result["confidence"],
                "total_latency_ms": total_latency,
                "metrics": {
                    "intent": self.metrics[-4] if len(self.metrics) >= 4 else None,
                    "table": self.metrics[-3] if len(self.metrics) >= 3 else None,
                    "prune": self.metrics[-2] if len(self.metrics) >= 2 else None,
                    "sql": self.metrics[-1] if len(self.metrics) >= 1 else None,
                }
            }
            
        except Exception as e:
            return {
                "question": question,
                "error": str(e),
                "total_latency_ms": (time.time() - pipeline_start) * 1000
            }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get aggregated metrics across all agent calls."""
        if not self.metrics:
            return {}
        
        by_agent = {}
        for metric in self.metrics:
            if metric.agent_name not in by_agent:
                by_agent[metric.agent_name] = []
            by_agent[metric.agent_name].append(metric)
        
        summary = {}
        for agent_name, metrics_list in by_agent.items():
            successful = [m for m in metrics_list if m.function_called]
            tool_calls_used = [m for m in successful if m.used_tool_calls is True]
            fallback_used = [m for m in successful if m.used_tool_calls is False]
            accurate_tool_calls = [m for m in successful if m.tool_call_accuracy is True]
            
            # Count validation errors
            total_validation_errors = sum(
                len(m.validation_errors) if m.validation_errors else 0 
                for m in successful
            )
            
            summary[agent_name] = {
                "total_calls": len(metrics_list),
                "successful_calls": len(successful),
                "success_rate": len(successful) / len(metrics_list) if metrics_list else 0,
                "avg_latency_ms": sum(m.latency_ms for m in successful) / len(successful) if successful else 0,
                "min_latency_ms": min((m.latency_ms for m in successful), default=0),
                "max_latency_ms": max((m.latency_ms for m in successful), default=0),
                "avg_tokens": sum(m.tokens_used for m in successful) / len(successful) if successful else 0,
                "total_tokens": sum(m.tokens_used for m in metrics_list),
                "tool_calls_usage": {
                    "structured_tool_calls": len(tool_calls_used),
                    "fallback_parsing": len(fallback_used),
                    "tool_calls_rate": len(tool_calls_used) / len(successful) if successful else 0
                },
                "tool_call_accuracy": {
                    "accurate_calls": len(accurate_tool_calls),
                    "accuracy_rate": len(accurate_tool_calls) / len(successful) if successful else 0,
                    "total_validation_errors": total_validation_errors,
                    "avg_validation_errors_per_call": total_validation_errors / len(successful) if successful else 0
                }
            }
        
        return summary

