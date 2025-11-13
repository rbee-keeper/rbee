# HuggingFace API Filters - Quick Reference

**TEAM-502** | **Date:** 2025-11-13

## Recommended Default Filters

### LLM Worker (`llm-worker-rbee`)
```
pipeline_tag=text-generation
library=transformers
filter=gguf,safetensors
```

**Example URL:**
```
https://huggingface.co/api/models?limit=50&pipeline_tag=text-generation&library=transformers&filter=gguf,safetensors
```

**Why:**
- LLM worker supports BOTH gguf and safetensors formats
- Filters out PyTorch-only models
- Returns 50+ compatible models

---

### SD Worker (`sd-worker-rbee`)
```
pipeline_tag=text-to-image
library=diffusers
filter=safetensors
```

**Example URL:**
```
https://huggingface.co/api/models?limit=50&pipeline_tag=text-to-image&library=diffusers&filter=safetensors
```

**Why:**
- SD worker ONLY supports safetensors format
- Filters out PyTorch/pickle files
- Returns 50+ compatible models

---

## Filter Syntax

### Comma-Separated = OR Logic
```
filter=gguf,safetensors  → Returns models with gguf OR safetensors
filter=safetensors       → Returns models with safetensors only
```

### Multiple Filters
```
filter=gguf,safetensors,apache-2.0  → Returns models with ANY of these tags
```

---

## Available Filter Tags

### Format Tags
- `gguf` - GGUF format (llama.cpp)
- `safetensors` - SafeTensors format
- `pytorch` - PyTorch weights
- `onnx` - ONNX format
- `tensorflow` - TensorFlow format

### License Tags
- `apache-2.0` - Apache 2.0 license
- `mit` - MIT license
- `llama3.1` - Llama 3.1 license
- `cc-by-4.0` - Creative Commons BY 4.0
- See full list in `types.ts`

### Other Useful Tags
- `conversational` - Chat models
- `text-generation-inference` - TGI compatible
- `endpoints_compatible` - HF Inference Endpoints compatible

---

## Testing Commands

### Test LLM Filter
```bash
curl -s "https://huggingface.co/api/models?limit=10&pipeline_tag=text-generation&library=transformers&filter=gguf,safetensors" | jq '.[0:3] | .[] | .id'
```

### Test SD Filter
```bash
curl -s "https://huggingface.co/api/models?limit=10&pipeline_tag=text-to-image&library=diffusers&filter=safetensors" | jq '.[0:3] | .[] | .id'
```

### Verify Format Tags
```bash
curl -s "https://huggingface.co/api/models?limit=5&pipeline_tag=text-generation&library=transformers&filter=gguf" | jq '.[] | {id, formats: [.tags[]? | select(. == "gguf" or . == "safetensors")]}'
```

---

## Implementation

### TypeScript Example
```typescript
import type { HuggingFaceListModelsParams } from '@rbee/marketplace-core'

// LLM models
const llmParams: HuggingFaceListModelsParams = {
  pipeline_tag: 'text-generation',
  library: 'transformers',
  filter: 'gguf,safetensors',
  limit: 50,
  sort: 'downloads',
  direction: -1,
}

// SD models
const sdParams: HuggingFaceListModelsParams = {
  pipeline_tag: 'text-to-image',
  library: 'diffusers',
  filter: 'safetensors',
  limit: 50,
  sort: 'downloads',
  direction: -1,
}
```

---

## Key Insights

✅ **API filtering is reliable** - No need for complex client-side filtering  
✅ **Format tags are consistent** - Models are properly tagged  
✅ **50+ compatible models** - Plenty of options for users  
✅ **OR logic is powerful** - `filter=gguf,safetensors` gets both formats  
✅ **Defense in depth** - Still validate on client side for safety  

---

## References

- Full Analysis: `.docs/TEAM_502_HUGGINGFACE_FILTER_ANALYSIS.md`
- Worker Catalog: `/bin/80-global-worker-catalog/src/data.ts`
- Type Definitions: `/frontend/packages/marketplace-core/src/adapters/huggingface/types.ts`
