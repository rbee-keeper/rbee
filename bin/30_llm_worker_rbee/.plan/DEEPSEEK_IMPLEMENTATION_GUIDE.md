# DeepSeek-R1 Implementation Guide

**Created by:** TEAM-481  
**Date:** 2025-11-12  
**Priority:** 🔥 P1 - HIGHEST PRIORITY  
**Status:** READY TO IMPLEMENT

## Why DeepSeek First?

- **Trending #1 on HuggingFace:** 421K+ downloads
- **High demand:** Most popular model right now
- **Candle support:** ✅ Full implementation available
- **Impact:** MASSIVE - immediate user value

---

## Reference Implementation

### Candle Example Location
```
/home/vince/Projects/rbee/reference/candle/candle-examples/examples/deepseekv2/
├── main.rs          # Example usage
└── README.md        # Documentation
```

### Candle Transformers Implementation
```
/home/vince/Projects/rbee/reference/candle/candle-transformers/src/models/
└── deepseek2.rs     # Model implementation (34,389 bytes)
```

---

## Implementation Steps

### Phase 1: Study Candle Implementation (Day 1, Morning)

#### 1.1 Read the candle example
```bash
cd /home/vince/Projects/rbee/reference/candle
cat candle-examples/examples/deepseekv2/main.rs
cat candle-examples/examples/deepseekv2/README.md
```

#### 1.2 Study the model implementation
```bash
cat candle-transformers/src/models/deepseek2.rs
```

**Key things to understand:**
- Model architecture (DeepSeekV2, DeepSeekV2Config)
- Forward pass signature
- KV cache management
- EOS token handling
- Vocab size

#### 1.3 Check for GGUF support
```bash
# Look for quantized version
find candle-transformers/src/models -name "*deepseek*" -o -name "*quantized*deepseek*"
```

---

### Phase 2: Create Safetensors Support (Day 1, Afternoon)

#### 2.1 Create `src/backend/models/deepseek.rs`

**Template structure:**
```rust
// TEAM-482: DeepSeek-R1 / DeepSeek-V2 support (safetensors)
//
// Created by: TEAM-482
// Reference: candle-transformers/src/models/deepseek2.rs

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::deepseek2::{DeepSeekV2, DeepSeekV2Config};
use std::path::Path;

pub struct DeepSeekModel {
    model: DeepSeekV2,
    eos_token_id: u32,
}

impl DeepSeekModel {
    /// Load DeepSeek model from safetensors
    pub fn load(model_path: &Path, device: &Device) -> Result<Self> {
        // 1. Load config.json
        let config_path = model_path.join("config.json");
        let config: DeepSeekV2Config = serde_json::from_reader(
            std::fs::File::open(&config_path)
                .with_context(|| format!("Failed to open config.json at {:?}", config_path))?,
        )
        .context("Failed to parse DeepSeek config.json")?;

        // 2. Find safetensors files
        let (model_dir, safetensor_files) = super::find_safetensors_files(model_path)?;

        // 3. Load model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&safetensor_files, candle_core::DType::F32, device)?
        };

        // 4. Build model
        let model = DeepSeekV2::load(vb, &config)?;

        // 5. Get EOS token ID from config
        let eos_token_id = config.eos_token_id.unwrap_or(2); // Default to 2 if not specified

        Ok(Self { model, eos_token_id })
    }

    /// Forward pass
    pub fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        self.model.forward(input_ids, position)
    }

    /// Get EOS token ID
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Get vocab size
    pub fn vocab_size(&self) -> usize {
        self.model.config().vocab_size
    }

    /// Reset KV cache
    pub fn reset_cache(&mut self) -> Result<()> {
        self.model.clear_kv_cache();
        Ok(())
    }
}
```

#### 2.2 Add to `src/backend/models/mod.rs`

**Step 1: Add module declaration**
```rust
pub mod deepseek;
```

**Step 2: Add to Model enum**
```rust
pub enum Model {
    // ... existing variants ...
    DeepSeek(deepseek::DeepSeekModel),
}
```

**Step 3: Add to forward() method**
```rust
pub fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
    match self {
        // ... existing variants ...
        Model::DeepSeek(m) => m.forward(input_ids, position),
    }
}
```

**Step 4: Add to eos_token_id() method**
```rust
pub fn eos_token_id(&self) -> u32 {
    match self {
        // ... existing variants ...
        Model::DeepSeek(m) => m.eos_token_id(),
    }
}
```

**Step 5: Add to architecture() method**
```rust
pub fn architecture(&self) -> &str {
    match self {
        // ... existing variants ...
        Model::DeepSeek(_) => "deepseek",
    }
}
```

**Step 6: Add to vocab_size() method**
```rust
pub fn vocab_size(&self) -> usize {
    match self {
        // ... existing variants ...
        Model::DeepSeek(m) => m.vocab_size(),
    }
}
```

**Step 7: Add to reset_cache() method**
```rust
pub fn reset_cache(&mut self) -> Result<()> {
    match self {
        // ... existing variants ...
        Model::DeepSeek(m) => m.reset_cache(),
    }
}
```

**Step 8: Add to detect_architecture() function**
```rust
pub fn detect_architecture(config_json: &Value) -> Result<String> {
    // ... existing checks ...
    
    // Check "architectures" array
    if let Some(archs) = config_json.get("architectures").and_then(|v| v.as_array()) {
        if let Some(arch) = archs.first().and_then(|v| v.as_str()) {
            let arch_lower = arch.to_lowercase();
            // ... existing checks ...
            if arch_lower.contains("deepseek") {
                return Ok("deepseek".to_string());
            }
            // ... rest of checks ...
        }
    }
    
    // ... rest of function ...
}
```

**Step 9: Add to load_model() function**
```rust
pub fn load_model(model_path: &str, device: &Device) -> Result<Model> {
    // ... existing GGUF check ...
    
    // Otherwise, load from safetensors with config.json
    let config_json = load_config_json(path)?;
    let architecture = detect_architecture(&config_json)?;

    match architecture.as_str() {
        // ... existing architectures ...
        "deepseek" => {
            let model = deepseek::DeepSeekModel::load(path, device)?;
            Ok(Model::DeepSeek(model))
        }
        // ... rest of architectures ...
    }
}
```

---

### Phase 3: Create GGUF Support (Day 2, Morning)

#### 3.1 Check if GGUF format exists for DeepSeek

**Research:**
- Check HuggingFace for DeepSeek GGUF models
- Check if candle has quantized_deepseek implementation
- If not, we may need to use quantized_llama (like we do for Mistral)

#### 3.2 Create `src/backend/models/quantized_deepseek.rs`

**If candle has native support:**
```rust
// TEAM-482: DeepSeek-R1 / DeepSeek-V2 GGUF support
//
// Created by: TEAM-482

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use std::path::Path;

pub struct QuantizedDeepSeekModel {
    model: candle_transformers::models::quantized_deepseek::DeepSeekV2, // If exists
    eos_token_id: u32,
}

impl QuantizedDeepSeekModel {
    pub fn load(gguf_path: &Path, device: &Device) -> Result<Self> {
        // Load GGUF file
        let mut file = std::fs::File::open(gguf_path)
            .with_context(|| format!("Failed to open GGUF file: {}", gguf_path.display()))?;
        
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .with_context(|| format!("Failed to read GGUF content from {}", gguf_path.display()))?;

        // Load model from GGUF
        let model = candle_transformers::models::quantized_deepseek::DeepSeekV2::from_gguf(content, &mut file, device)?;

        // Get EOS token from metadata
        let eos_token_id = /* extract from GGUF metadata */ 2;

        Ok(Self { model, eos_token_id })
    }

    pub fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        self.model.forward(input_ids, position)
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub fn vocab_size(&self) -> usize {
        self.model.vocab_size()
    }

    pub fn reset_cache(&mut self) -> Result<()> {
        self.model.clear_kv_cache();
        Ok(())
    }
}
```

**If no native support (fallback to quantized_llama):**
```rust
// TEAM-482: DeepSeek GGUF uses quantized_llama loader
// (Similar to how Mistral GGUF uses quantized_llama)

pub use super::quantized_llama::QuantizedLlamaModel as QuantizedDeepSeekModel;
```

#### 3.3 Add GGUF support to `mod.rs`

**Step 1: Add module declaration**
```rust
pub mod quantized_deepseek;
```

**Step 2: Add to Model enum**
```rust
pub enum Model {
    // ... existing variants ...
    QuantizedDeepSeek(quantized_deepseek::QuantizedDeepSeekModel),
}
```

**Step 3: Add to all match statements** (forward, eos_token_id, etc.)

**Step 4: Add to detect_architecture_from_gguf()**
```rust
fn detect_architecture_from_gguf(gguf_path: &Path) -> Result<String> {
    // ... existing code ...
    
    let arch = content
        .metadata
        .get("general.architecture")
        .and_then(|v| match v {
            candle_core::quantized::gguf_file::Value::String(s) => Some(s.clone()),
            _ => None,
        })
        .context("Missing general.architecture in GGUF metadata")?;

    Ok(arch)
}
```

**Step 5: Add to GGUF loading in load_model()**
```rust
if model_path.ends_with(".gguf") {
    let architecture = detect_architecture_from_gguf(path)?;

    match architecture.as_str() {
        // ... existing architectures ...
        "deepseek" => {
            let model = quantized_deepseek::QuantizedDeepSeekModel::load(path, device)?;
            Ok(Model::QuantizedDeepSeek(model))
        }
        // ... rest of architectures ...
    }
}
```

---

### Phase 4: Testing (Day 2, Afternoon)

#### 4.1 Download test models

**Safetensors:**
```bash
# Download DeepSeek-R1 or DeepSeek-V2 from HuggingFace
# Example: deepseek-ai/DeepSeek-R1
```

**GGUF:**
```bash
# Download quantized version
# Example: TheBloke/DeepSeek-R1-GGUF
```

#### 4.2 Test safetensors loading
```bash
cargo run --bin llm-worker -- \
  --model-path /path/to/deepseek-r1 \
  --device cuda
```

#### 4.3 Test GGUF loading
```bash
cargo run --bin llm-worker -- \
  --model-path /path/to/deepseek-r1.Q4_K_M.gguf \
  --device cuda
```

#### 4.4 Test inference
```bash
curl -X POST http://localhost:7833/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

#### 4.5 Test streaming
```bash
curl -X POST http://localhost:7833/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "stream": true
  }'
```

---

### Phase 5: Documentation (Day 3, Morning)

#### 5.1 Update README
```markdown
## Supported Models

- ✅ Llama (safetensors + GGUF)
- ✅ Mistral (safetensors + GGUF)
- ✅ Phi (safetensors + GGUF)
- ✅ Qwen (safetensors + GGUF)
- ✅ Gemma (GGUF only)
- ✅ DeepSeek (safetensors + GGUF) ← NEW
```

#### 5.2 Add example usage
```markdown
### DeepSeek-R1 Example

**Safetensors:**
```bash
llm-worker --model-path /models/deepseek-r1 --device cuda
```

**GGUF:**
```bash
llm-worker --model-path /models/deepseek-r1.Q4_K_M.gguf --device cuda
```
```

#### 5.3 Update CHANGELOG
```markdown
## [Unreleased]

### Added
- DeepSeek-R1 / DeepSeek-V2 support (safetensors + GGUF) - TEAM-482
  - Trending #1 on HuggingFace (421K+ downloads)
  - Full support for both safetensors and GGUF formats
```

---

## Verification Checklist

- [ ] Safetensors loading works
- [ ] GGUF loading works
- [ ] Inference produces correct output
- [ ] Streaming works
- [ ] Cache reset works
- [ ] EOS token handling correct
- [ ] No memory leaks
- [ ] Performance acceptable
- [ ] Documentation updated
- [ ] Tests passing

---

## Common Issues & Solutions

### Issue 1: Config.json missing fields
**Solution:** Check candle example for default values

### Issue 2: GGUF architecture not detected
**Solution:** Check `general.architecture` field in GGUF metadata

### Issue 3: Forward pass signature mismatch
**Solution:** Check if DeepSeek uses position parameter (most do)

### Issue 4: KV cache not clearing
**Solution:** Check if model has `clear_kv_cache()` method

---

## Success Criteria

- ✅ DeepSeek-R1 loads successfully (safetensors)
- ✅ DeepSeek-R1 loads successfully (GGUF)
- ✅ Inference works correctly
- ✅ Streaming works correctly
- ✅ No regression in existing models
- ✅ Documentation complete

---

## Estimated Effort

- **Day 1:** Study + Safetensors implementation (6-8 hours)
- **Day 2:** GGUF implementation + Testing (6-8 hours)
- **Day 3:** Documentation + Polish (2-4 hours)

**Total:** 2-3 days

---

## Next Steps After DeepSeek

1. **TEAM-483:** Add Gemma safetensors support (1-2 days)
2. **TEAM-484:** Implement Mixtral MoE support (2-3 days)

---

**Status:** ✅ READY TO IMPLEMENT  
**Priority:** 🔥 HIGHEST  
**Expected Impact:** MASSIVE (trending #1 model)
