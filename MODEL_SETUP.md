# Fireworks Model Setup

## Finding Available Models

The model names in the code are defaults. You need to check which models are available in your Fireworks account:

1. Go to https://app.fireworks.ai/models
2. Check which models you have access to
3. Update the model names in your code or environment variables

## Setting Model Names

### Option 1: Environment Variables (Recommended)

```bash
export FIREWORKS_API_KEY='your-api-key'
export FIREWORKS_MODEL='your-model-name-here'
export FIREWORKS_INTENT_MODEL='your-intent-model-name-here'
```

### Option 2: Code Configuration

```python
from fireworks_querygpt import FireworksQueryGPT

querygpt = FireworksQueryGPT(
    model="your-model-name-here",
    intent_model="your-intent-model-name-here"
)
```

## Common Model Names

Based on Fireworks documentation, try these formats:

- `accounts/fireworks/models/firefunction-v2` (for function calling)
- `accounts/fireworks/models/llama-v3-70b-instruct`
- `accounts/fireworks/models/llama-v3-8b-instruct`
- `accounts/fireworks/models/llama-v3.1-70b-instruct`

Or check your dashboard for the exact model names available to your account.

## Testing Models

Run the test script to find working models:

```bash
source venv_querygpt/bin/activate
export FIREWORKS_API_KEY='your-key'
python3 test_fireworks_models.py
```

## Troubleshooting

If you get "Model not found" errors:

1. **Check your API key** - Make sure it's valid and has access to models
2. **Check model availability** - Some models may require deployment first
3. **Check model names** - Use exact names from your Fireworks dashboard
4. **Check account permissions** - Some models may require specific account tiers

## Recommended Models

For this QueryGPT system:

- **Intent Agent**: Smaller, faster model (8B or similar) - classification task
- **Table/Column/SQL Agents**: Larger model (70B or FireFunction) - reasoning tasks

FireFunction models are optimized for function calling, which is perfect for our structured outputs.

