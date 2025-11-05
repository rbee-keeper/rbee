# TEAM-409: Phase 3 - Compatibility Matrix Data Layer

**Created:** 2025-11-05  
**Team:** TEAM-409  
**Duration:** 3-4 days  
**Status:** ✅ READY (TEAM-408 complete)  
**Dependencies:** ✅ TEAM-408 complete (worker catalog SDK)

---

## 🎯 Mission

**PRIMARY GOAL:** Filter HuggingFace models so we ONLY show models that our LLM workers can run.

Implement the compatibility matrix to:
1. **Filter HuggingFace API results** - Only show compatible models in marketplace
2. **Check model→worker compatibility** - Determine if a specific model can run on our workers
3. **Prevent advertising incompatible models** - Don't generate static pages for models we can't run

**Key Principle:** "If we don't support it, it doesn't exist" (from TEAM-406 research)

---

## ✅ Checklist

### Task 3.1: Create Compatibility Module (Rust)
- [ ] Create `bin/79_marketplace_core/marketplace-sdk/src/compatibility.rs`
- [ ] Define `CompatibilityResult` struct
- [ ] Define `CompatibilityReason` enum
- [ ] Implement `is_model_compatible(model_metadata)` - Check if model is compatible with ANY worker
- [ ] Implement `check_model_worker_compatibility(model, worker)` - Check specific model-worker pair
- [ ] Add TEAM-409 signatures
- [ ] Commit: "TEAM-409: Add compatibility module"

**Types:**
```rust
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct CompatibilityResult {
    pub compatible: bool,
    pub confidence: CompatibilityConfidence,
    pub reasons: Vec<String>,
    pub warnings: Vec<String>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "lowercase")]
pub enum CompatibilityConfidence {
    High,      // Tested and verified
    Medium,    // Should work based on specs
    Low,       // Might work, untested
    None,      // Incompatible
}

pub fn check_compatibility(
    model: &ModelMetadata,
    worker: &WorkerBinary,
) -> CompatibilityResult {
    let mut reasons = Vec::new();
    let mut warnings = Vec::new();
    let mut recommendations = Vec::new();
    
    // Check architecture
    if !worker.supported_architectures.contains(&model.architecture.to_string()) {
        reasons.push(format!(
            "Worker does not support {} architecture",
            model.architecture
        ));
        return CompatibilityResult {
            compatible: false,
            confidence: CompatibilityConfidence::None,
            reasons,
            warnings,
            recommendations,
        };
    }
    
    // Check format
    if !worker.supported_formats.contains(&model.format.to_string()) {
        reasons.push(format!(
            "Worker does not support {} format",
            model.format
        ));
        return CompatibilityResult {
            compatible: false,
            confidence: CompatibilityConfidence::None,
            reasons,
            warnings,
            recommendations,
        };
    }
    
    // Check context length
    if model.max_context_length > worker.max_context_length {
        warnings.push(format!(
            "Model context ({}) exceeds worker limit ({})",
            model.max_context_length,
            worker.max_context_length
        ));
    }
    
    // Determine confidence based on testing status
    let confidence = match (model.architecture, worker.worker_type) {
        (ModelArchitecture::Llama, _) => CompatibilityConfidence::High,
        (ModelArchitecture::Mistral, _) => CompatibilityConfidence::Medium,
        (ModelArchitecture::Phi, _) => CompatibilityConfidence::Medium,
        (ModelArchitecture::Qwen, _) => CompatibilityConfidence::Medium,
        _ => CompatibilityConfidence::Low,
    };
    
    reasons.push("Architecture and format compatible".to_string());
    
    CompatibilityResult {
        compatible: true,
        confidence,
        reasons,
        warnings,
        recommendations,
    }
}
```

**Acceptance:**
- ✅ Compatibility module compiles
- ✅ Check function handles all cases
- ✅ Returns detailed results

---

### Task 3.2: Add Model Metadata Extraction
- [ ] Create `bin/79_marketplace_core/marketplace-sdk/src/model_metadata.rs`
- [ ] Implement `extract_metadata_from_hf(model_id)` function
- [ ] Parse HuggingFace model card
- [ ] Extract architecture from tags/config
- [ ] Extract format from files
- [ ] Extract quantization from filename
- [ ] Add error handling
- [ ] Commit: "TEAM-409: Add model metadata extraction"

**Implementation:**
```rust
pub async fn extract_metadata_from_hf(
    model_id: &str,
) -> Result<ModelMetadata, MetadataError> {
    // Fetch model info from HuggingFace API
    let model_info = fetch_hf_model_info(model_id).await?;
    
    // Extract architecture from tags
    let architecture = detect_architecture(&model_info.tags)?;
    
    // Extract format from files
    let format = detect_format(&model_info.siblings)?;
    
    // Extract quantization from filename
    let quantization = detect_quantization(model_id);
    
    // Extract context length from config
    let max_context_length = extract_context_length(&model_info).await?;
    
    Ok(ModelMetadata {
        architecture,
        format,
        quantization,
        parameters: extract_parameter_count(&model_info),
        size_bytes: calculate_size(&model_info.siblings),
        max_context_length,
    })
}

fn detect_architecture(tags: &[String]) -> Result<ModelArchitecture, MetadataError> {
    if tags.iter().any(|t| t.contains("llama")) {
        Ok(ModelArchitecture::Llama)
    } else if tags.iter().any(|t| t.contains("mistral")) {
        Ok(ModelArchitecture::Mistral)
    } else if tags.iter().any(|t| t.contains("phi")) {
        Ok(ModelArchitecture::Phi)
    } else if tags.iter().any(|t| t.contains("qwen")) {
        Ok(ModelArchitecture::Qwen)
    } else if tags.iter().any(|t| t.contains("gemma")) {
        Ok(ModelArchitecture::Gemma)
    } else {
        Ok(ModelArchitecture::Unknown)
    }
}

fn detect_format(files: &[HfFile]) -> Result<ModelFormat, MetadataError> {
    if files.iter().any(|f| f.filename.ends_with(".safetensors")) {
        Ok(ModelFormat::SafeTensors)
    } else if files.iter().any(|f| f.filename.ends_with(".gguf")) {
        Ok(ModelFormat::Gguf)
    } else if files.iter().any(|f| f.filename.ends_with(".bin")) {
        Ok(ModelFormat::Pytorch)
    } else {
        Err(MetadataError::UnknownFormat)
    }
}
```

**Acceptance:**
- ✅ Can extract metadata from HuggingFace
- ✅ Handles various model formats
- ✅ Error handling for missing data

---

### Task 3.3: Add HuggingFace Filter Function
- [ ] Add `filter_compatible_models(models)` to compatibility.rs
- [ ] Filter models by supported architectures (llama, mistral, phi, qwen, gemma)
- [ ] Filter models by supported formats (safetensors, gguf)
- [ ] Filter models by quantization support
- [ ] Return only models compatible with at least ONE worker
- [ ] Add TEAM-409 signatures
- [ ] Commit: "TEAM-409: Add HuggingFace filter function"

**Implementation:**
```rust
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct CompatibilityMatrix {
    pub models: Vec<String>,  // Model IDs
    pub workers: Vec<String>, // Worker IDs
    pub matrix: Vec<Vec<CompatibilityResult>>,
}

pub fn generate_compatibility_matrix(
    models: &[ModelMetadata],
    workers: &[WorkerBinary],
) -> CompatibilityMatrix {
    let mut matrix = Vec::new();
    
    for model in models {
        let mut row = Vec::new();
        for worker in workers {
            let result = check_compatibility(model, worker);
            row.push(result);
        }
        matrix.push(row);
    }
    
    CompatibilityMatrix {
        models: models.iter().map(|m| m.id.clone()).collect(),
        workers: workers.iter().map(|w| w.id.clone()).collect(),
        matrix,
    }
}
```

**Acceptance:**
- ✅ Matrix generator works
- ✅ Handles large datasets efficiently
- ✅ Results cacheable

---

### Task 3.4: Add WASM Bindings for Compatibility
- [ ] Update `bin/79_marketplace_core/marketplace-sdk/src/wasm_worker.rs`
- [ ] Add `#[wasm_bindgen]` for `is_model_compatible()`
- [ ] Add `#[wasm_bindgen]` for `check_model_worker_compatibility()`
- [ ] Add `#[wasm_bindgen]` for `filter_compatible_models()`
- [ ] Export from lib.rs
- [ ] Commit: "TEAM-409: Add WASM compatibility bindings"

**Example:**
```rust
#[wasm_bindgen]
pub fn check_model_worker_compatibility(
    model: JsValue,
    worker: JsValue,
) -> Result<JsValue, JsValue> {
    let model: ModelMetadata = serde_wasm_bindgen::from_value(model)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let worker: WorkerBinary = serde_wasm_bindgen::from_value(worker)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    let result = check_compatibility(&model, &worker);
    
    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}
```

**Acceptance:**
- ✅ WASM functions compile
- ✅ Can call from JavaScript
- ✅ Types serialize correctly

---

### Task 3.5: Update marketplace-node with Compatibility Functions
- [ ] Open `frontend/packages/marketplace-node/src/index.ts`
- [ ] Add `isModelCompatible(modelMetadata)` function - Check if model can run on ANY worker
- [ ] Add `filterCompatibleModels(models)` function - Filter HuggingFace results
- [ ] Add `checkModelWorkerCompatibility(model, worker)` function - Check specific pair
- [ ] Call WASM functions
- [ ] Add TypeScript types
- [ ] Commit: "TEAM-409: Add compatibility functions to marketplace-node"

**Implementation:**
```typescript
export async function checkCompatibility(
  model: ModelMetadata,
  worker: Worker
): Promise<CompatibilityResult> {
  const sdk = await getSDK()
  return await sdk.check_model_worker_compatibility(model, worker)
}

export async function getCompatibleWorkersForModel(
  modelId: string
): Promise<Worker[]> {
  const sdk = await getSDK()
  
  // Extract model metadata
  const metadata = await sdk.extract_metadata_from_hf(modelId)
  
  // Get all workers
  const workers = await listWorkerBinaries()
  
  // Filter compatible workers
  const compatible = []
  for (const worker of workers) {
    const result = await checkCompatibility(metadata, worker)
    if (result.compatible) {
      compatible.push(worker)
    }
  }
  
  return compatible
}

export async function getCompatibleModelsForWorker(
  workerId: string,
  models: ModelMetadata[]
): Promise<ModelMetadata[]> {
  const sdk = await getSDK()
  
  // Get worker
  const worker = await sdk.get_worker(workerId)
  if (!worker) return []
  
  // Filter compatible models
  const compatible = []
  for (const model of models) {
    const result = await checkCompatibility(model, worker)
    if (result.compatible) {
      compatible.push(model)
    }
  }
  
  return compatible
}
```

**Acceptance:**
- ✅ Functions call WASM successfully
- ✅ Return correct compatibility results
- ✅ TypeScript types correct

---

### Task 3.6: Update HuggingFace Integration
- [ ] Open `frontend/packages/marketplace-node/src/huggingface.ts`
- [ ] Update `listHuggingFaceModels()` to filter by compatibility
- [ ] Add `onlyCompatible` parameter (default: true)
- [ ] Call `filterCompatibleModels()` on results
- [ ] Update `searchHuggingFaceModels()` to filter
- [ ] Commit: "TEAM-409: Filter HuggingFace models by compatibility"

**Implementation:**
```rust
use std::collections::HashMap;
use std::time::{Duration, Instant};

pub struct CompatibilityCache {
    cache: HashMap<String, CacheEntry>,
    ttl: Duration,
}

struct CacheEntry {
    result: CompatibilityResult,
    inserted_at: Instant,
}

impl CompatibilityCache {
    pub fn new(ttl_seconds: u64) -> Self {
        Self {
            cache: HashMap::new(),
            ttl: Duration::from_secs(ttl_seconds),
        }
    }
    
    pub fn get(&self, model_id: &str, worker_id: &str) -> Option<&CompatibilityResult> {
        let key = format!("{}:{}", model_id, worker_id);
        self.cache.get(&key).and_then(|entry| {
            if entry.inserted_at.elapsed() < self.ttl {
                Some(&entry.result)
            } else {
                None
            }
        })
    }
    
    pub fn insert(&mut self, model_id: &str, worker_id: &str, result: CompatibilityResult) {
        let key = format!("{}:{}", model_id, worker_id);
        self.cache.insert(key, CacheEntry {
            result,
            inserted_at: Instant::now(),
        });
    }
}
```

**Acceptance:**
- ✅ Cache reduces redundant checks
- ✅ TTL prevents stale data
- ✅ Performance improvement measurable

---

### Task 3.7: Write Unit Tests
- [ ] Create `bin/79_marketplace_core/marketplace-sdk/tests/compatibility_tests.rs`
- [ ] Test `check_compatibility()` with various scenarios
- [ ] Test metadata extraction
- [ ] Test matrix generation
- [ ] Test cache functionality
- [ ] Run `cargo test -p marketplace-sdk`
- [ ] Commit: "TEAM-409: Add compatibility tests"

**Test Cases:**
```rust
#[test]
fn test_compatible_llama_cpu() {
    let model = ModelMetadata {
        architecture: ModelArchitecture::Llama,
        format: ModelFormat::SafeTensors,
        max_context_length: 4096,
        ..Default::default()
    };
    
    let worker = WorkerBinary {
        worker_type: WorkerType::Cpu,
        supported_architectures: vec!["llama".to_string()],
        supported_formats: vec!["safetensors".to_string()],
        max_context_length: 8192,
        ..Default::default()
    };
    
    let result = check_compatibility(&model, &worker);
    assert!(result.compatible);
    assert_eq!(result.confidence, CompatibilityConfidence::High);
}

#[test]
fn test_incompatible_format() {
    let model = ModelMetadata {
        architecture: ModelArchitecture::Llama,
        format: ModelFormat::Gguf,
        ..Default::default()
    };
    
    let worker = WorkerBinary {
        supported_formats: vec!["safetensors".to_string()],
        ..Default::default()
    };
    
    let result = check_compatibility(&model, &worker);
    assert!(!result.compatible);
    assert!(result.reasons.iter().any(|r| r.contains("format")));
}
```

**Acceptance:**
- ✅ All tests pass
- ✅ Edge cases covered
- ✅ Error cases tested

---

### Task 3.8: Write Integration Tests
- [ ] Create `frontend/packages/marketplace-node/tests/compatibility.test.ts`
- [ ] Test `isModelCompatible()` with various models
- [ ] Test `filterCompatibleModels()` filters correctly
- [ ] Test `checkModelWorkerCompatibility()` for specific pairs
- [ ] Test HuggingFace integration filters incompatible models
- [ ] Run `pnpm test`
- [ ] Commit: "TEAM-409: Add compatibility integration tests"

**Test Setup:**
```typescript
import { describe, it, expect } from 'vitest'
import { checkCompatibility, getCompatibleWorkersForModel } from '../src/index'

describe('Compatibility Matrix', () => {
  it('should check compatibility', async () => {
    const model = {
      architecture: 'llama',
      format: 'safetensors',
      max_context_length: 4096,
    }
    
    const worker = {
      worker_type: 'cpu',
      supported_architectures: ['llama'],
      supported_formats: ['safetensors'],
      max_context_length: 8192,
    }
    
    const result = await checkCompatibility(model, worker)
    expect(result.compatible).toBe(true)
  })
  
  it('should get compatible workers for model', async () => {
    const workers = await getCompatibleWorkersForModel('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    expect(workers.length).toBeGreaterThan(0)
  })
})
```

**Acceptance:**
- ✅ All tests pass
- ✅ Real API calls work
- ✅ Results accurate

---

### Task 3.9: Update Documentation
- [ ] Update `bin/79_marketplace_core/marketplace-sdk/README.md`
- [ ] Add compatibility check examples
- [ ] Document metadata extraction
- [ ] Add API reference
- [ ] Update `frontend/packages/marketplace-node/README.md`
- [ ] Add compatibility examples
- [ ] Commit: "TEAM-409: Update compatibility documentation"

**README Example:**
```markdown
## Compatibility Checks

### Check Model-Worker Compatibility
\`\`\`typescript
import { checkCompatibility } from '@rbee/marketplace-node'

const model = { architecture: 'llama', format: 'safetensors', ... }
const worker = { worker_type: 'cpu', supported_architectures: ['llama'], ... }

const result = await checkCompatibility(model, worker)
console.log(result.compatible) // true/false
console.log(result.reasons)    // ["Architecture and format compatible"]
console.log(result.warnings)   // []
\`\`\`

### Get Compatible Workers
\`\`\`typescript
import { getCompatibleWorkersForModel } from '@rbee/marketplace-node'

const workers = await getCompatibleWorkersForModel('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
console.log(workers) // [{ id: 'llm-worker-rbee-cpu', ... }]
\`\`\`
```

**Acceptance:**
- ✅ README updated
- ✅ Examples clear
- ✅ API documented

---

### Task 3.10: Verification
- [ ] Run `cargo check --workspace` - ZERO errors
- [ ] Run `cargo test -p marketplace-sdk` - ALL PASS
- [ ] Run `pnpm test` in marketplace-node - ALL PASS
- [ ] Build WASM: `wasm-pack build --target bundler`
- [ ] Test compatibility checks in Node.js
- [ ] Review all changes for TEAM-409 signatures
- [ ] Create handoff document (max 2 pages)

**Handoff Document Contents:**
- Compatibility module implemented
- Metadata extraction working
- Matrix generation functional
- Test coverage
- Next team ready: TEAM-410

---

## 📁 Files Created/Modified

### New Files
- `bin/79_marketplace_core/marketplace-sdk/src/compatibility.rs` - Compatibility checking logic
- `bin/79_marketplace_core/marketplace-sdk/src/model_metadata.rs` - Model metadata extraction
- `bin/79_marketplace_core/marketplace-sdk/tests/compatibility_tests.rs` - Unit tests
- `frontend/packages/marketplace-node/tests/compatibility.test.ts` - Integration tests
- `bin/.plan/TEAM_409_COMPATIBILITY_HANDOFF.md` - Handoff document

### Modified Files
- `bin/79_marketplace_core/marketplace-sdk/src/lib.rs` - Re-exports
- `bin/79_marketplace_core/marketplace-sdk/src/wasm_worker.rs` - WASM bindings
- `frontend/packages/marketplace-node/src/index.ts` - Compatibility functions
- `frontend/packages/marketplace-node/src/huggingface.ts` - Filter by compatibility
- `bin/79_marketplace_core/marketplace-sdk/README.md` - Documentation
- `frontend/packages/marketplace-node/README.md` - Documentation

---

## ⚠️ Blockers & Dependencies

### Blocked By
- TEAM-408 (needs worker catalog SDK)

### Blocks
- TEAM-410 (needs compatibility data for Next.js integration)

---

## 🎯 Success Criteria

- [ ] **PRIMARY:** HuggingFace models filtered by compatibility
- [ ] `isModelCompatible()` function working
- [ ] `filterCompatibleModels()` function working
- [ ] Model metadata extraction working
- [ ] All tests passing (Rust + TypeScript)
- [ ] Documentation complete
- [ ] Marketplace only shows compatible models
- [ ] Handoff document complete (≤2 pages)

---

## 📚 References

- Engineering Rules: `.windsurf/rules/engineering-rules.md`
- Model support: `bin/30_llm_worker_rbee/docs/MODEL_SUPPORT.md`
- Worker catalog: `bin/80-hono-worker-catalog/src/data.ts`
- marketplace-sdk: `bin/99_shared_crates/marketplace-sdk/`

---

**TEAM-409 - Phase 3 Checklist v1.0**  
**Next Phase:** TEAM-410 (Next.js Integration)
