# Quick Start Guide

## Your API Key is Set

Your Fireworks API key has been configured: `fw_3ZahrBVt66YEAEbfUAAnXwbN`

## Next Steps

### 1. Check Available Models

Go to https://app.fireworks.ai/models and see which models you have access to.

### 2. Set Model Names (if needed)

If the default model names don't work, set them:

```bash
export FIREWORKS_MODEL='your-model-name-from-dashboard'
export FIREWORKS_INTENT_MODEL='your-intent-model-name'
```

### 3. Activate Virtual Environment

```bash
source venv_querygpt/bin/activate
```

### 4. Test the System

```bash
# Test with a simple question
python3 -c "
from fireworks_querygpt import FireworksQueryGPT
import os
os.environ['FIREWORKS_API_KEY'] = 'fw_3ZahrBVt66YEAEbfUAAnXwbN'
qgpt = FireworksQueryGPT()
result = qgpt.intent_agent('How many trips?')
print(f'Workspace: {result[\"workspace_id\"]}')
"
```

### 5. Run Full Example

```bash
export FIREWORKS_API_KEY='fw_3ZahrBVt66YEAEbfUAAnXwbN'
python3 example_querygpt_usage.py
```

## Troubleshooting

If you get "Model not found" errors:
1. Check https://app.fireworks.ai/models for available models
2. Update model names in environment variables or code
3. Some models may need to be deployed first in your dashboard

See `MODEL_SETUP.md` for more details.
