# Tool Calling Explained

## What Tool Calling SHOULD Be Doing

When using the OpenAI-compatible API with function calling, here's what should happen:

### Expected Behavior

1. **Request Format:**
   ```python
   response = client.chat.completions.create(
       model="model-name",
       messages=[{"role": "user", "content": "..."}],
       tools=[{
           "type": "function",
           "function": {
               "name": "classify_intent",
               "description": "...",
               "parameters": {...}  # JSON schema
           }
       }],
       tool_choice={"type": "function", "function": {"name": "classify_intent"}}
   )
   ```

2. **Expected Response Structure:**
   ```python
   response.choices[0].message.tool_calls = [
       {
           "id": "call_abc123",
           "type": "function",
           "function": {
               "name": "classify_intent",
               "arguments": '{"workspace_id": "mobility", "confidence": 0.9, "reasoning": "..."}'
           }
       }
   ]
   response.choices[0].message.content = None  # Should be None when tool_calls present
   ```

3. **What This Means:**
   - `message.tool_calls` is a **list** of tool call objects
   - Each tool call has a `function` object with `name` and `arguments` (JSON string)
   - `message.content` should be `None` or empty when tool calls are present
   - The `arguments` field contains a JSON string that needs to be parsed

### Why True Tool Calling Matters

1. **Structured Outputs**: Guarantees the model returns data matching your schema
2. **Type Safety**: The API validates types before returning
3. **Enum Validation**: Ensures enum values are valid
4. **Reliability**: Less parsing errors, more predictable behavior
5. **Performance**: Some providers optimize tool calling paths

## Current Issue: Fallback Parsing

### What's Happening Now

Instead of structured `tool_calls`, the models are returning:

```python
response.choices[0].message.tool_calls = None  # ❌ Should be a list
response.choices[0].message.content = '{"type": "function", "name": "classify_intent", "parameters": {...}}'
```

This means:
- The model is returning function call data as **text** in `content`
- We have to parse JSON manually
- No schema validation by the API
- More error-prone

### Possible Reasons

1. **Model Doesn't Support Tool Calling**: Some models may not fully support the tool calling API
2. **API Configuration**: The API endpoint might not be configured correctly
3. **Model Behavior**: The model might be trained to return function calls as text
4. **Provider Implementation**: Different providers may implement tool calling differently

## Tool Call Accuracy Metrics

We now measure:

1. **Tool Calls Usage Rate**: % of calls using structured `tool_calls` vs fallback
2. **Tool Call Accuracy**: % of calls that pass schema validation
3. **Validation Errors**: Count of schema violations (missing fields, invalid enums, wrong types)

### Validation Checks

For each agent, we validate:

- **Intent Agent**:
  - Required fields: `workspace_id`, `confidence`, `reasoning`
  - `workspace_id` is in allowed enum values
  - `confidence` is a number between 0 and 1
  - `reasoning` is a string

- **Table Agent**:
  - Required fields: `relevant_tables`, `reasoning`
  - `relevant_tables` is a list
  - Tables exist in workspace
  - `reasoning` is a string

- **Column Prune Agent**:
  - Required fields: `relevant_columns`, `irrelevant_columns`, `reasoning`
  - Columns are lists
  - Columns exist in table schema

- **SQL Generation Agent**:
  - Required fields: `sql_query`, `explanation`, `tables_used`
  - `sql_query` is a string
  - `tables_used` is a list of strings

## How to Fix True Tool Calling

1. **Check Model Support**: Verify the model supports tool calling
2. **Try Different Models**: Some models have better tool calling support
3. **Check API Version**: Ensure you're using the correct API endpoint
4. **Review Provider Docs**: Check Fireworks/OpenAI docs for tool calling requirements

## Comparison: Fireworks vs OpenAI

The evaluation now tracks:
- Which provider uses structured `tool_calls` more often
- Which provider has higher tool call accuracy
- Which provider has fewer validation errors

This helps identify which provider has better tool calling implementation.

