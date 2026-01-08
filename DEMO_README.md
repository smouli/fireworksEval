# Fireworks Features Demo

Demonstrations of Fireworks AI features: Prompt Caching and PII Protection.

## Quick Start

```bash
# Set API key
export FIREWORKS_API_KEY='your-api-key'

# Run all demos
python3 demo_fireworks_features.py

# Or run individually
python3 demo_prompt_caching.py
python3 demo_pii_protection.py
```

## 1. Prompt Caching Demo

**What it shows:**
- How Fireworks automatically caches common prompt prefixes
- Up to 80% reduction in Time To First Token (TTFT)
- Optimal prompt structure for maximum cache hits

**Key Points:**
- ✅ Enabled by default - no configuration needed
- ✅ Static content (system prompts, schemas) should be at the BEGINNING
- ✅ Variable content (user questions) should be at the END
- ✅ Automatically reduces latency and costs for repeated queries

**Usage:**
```bash
python3 demo_prompt_caching.py
```

## 2. PII Protection Demo

**What it shows:**
- Zero Data Retention policy (default)
- Encryption at rest and in transit
- SOC 2 Type II and HIPAA compliance
- How sensitive data is protected

**Key Points:**
- ✅ Zero data retention by default (no opt-in required)
- ✅ No logging or storage of prompts/generations
- ✅ Data exists only in volatile memory during request
- ✅ SOC 2 Type II and HIPAA compliant
- ✅ Suitable for healthcare, finance, and regulated industries

**Usage:**
```bash
python3 demo_pii_protection.py
```

## 3. Combined Demo

Run both demos together:
```bash
python3 demo_fireworks_features.py
```

## Integration with QueryGPT

The QueryGPT implementation (`fireworks_querygpt.py`) is already optimized for prompt caching:

- **Static content first**: System prompts, workspace descriptions, table schemas
- **Variable content last**: User questions, specific filters

This structure maximizes cache hits and reduces latency automatically.

## Benefits

### Prompt Caching
- **Performance**: Up to 80% faster TTFT for cached prompts
- **Cost**: Reduced token processing for repeated prefixes
- **Scalability**: Better performance under high load

### PII Protection
- **Security**: Zero data retention ensures no sensitive data is stored
- **Compliance**: SOC 2 Type II and HIPAA certified
- **Trust**: Suitable for regulated industries without additional setup

## Technical Details

### Prompt Caching
- Enabled automatically for all Fireworks models
- Caches internal model state for common prefixes
- No API changes needed - just structure prompts optimally
- Works with streaming and non-streaming requests

### PII Protection
- Zero Data Retention: Default policy, no data logged
- Encryption: TLS in transit, AES at rest
- Compliance: SOC 2 Type II, HIPAA, GDPR-ready
- Audit: No persistent logs of sensitive information

## Next Steps

1. Run the demos to see the features in action
2. Review how QueryGPT uses prompt caching optimization
3. Understand PII protection for compliance requirements
4. Integrate these features into your production workflows

