# LLM Worker Candle Idiom Audit

**Date:** 2025-11-03  
**Question:** Is the LLM worker Candle idiomatic?

---

## ✅ VERDICT: YES, Mostly Idiomatic (8/10)

The LLM worker is **significantly more Candle idiomatic** than the old SD worker was.

---

## ✅ What's Candle Idiomatic

### 1. Model Enum Pattern ✅
**File:** `backend/models/mod.rs`

```rust
pub enum Model {
    Llama(llama::LlamaModel),
    QuantizedLlama(quantized_llama::QuantizedLlamaModel),
    Mistral(mistral::MistralModel),
    Phi(phi::PhiModel),
    QuantizedPhi(quantized_phi::QuantizedPhiModel),
    Qwen(qwen::QwenModel),
    QuantizedQwen(quantized_qwen::QuantizedQwenModel),
}
```

**Assessment:** ✅ **EXCELLENT** - This is the Candle-idiomatic enum pattern
- Uses enum for polymorphism (not traits)
- Each variant wraps a specific Candle model type
- Delegates to natural model interfaces

**Comment from code:**
> "TEAM-017: Each variant wraps a specific model type with its natural interface"
> "TEAM-017: Refactored by: TEAM-017 (switched to enum pattern for Candle idiomaticity)"

### 2. Direct Candle Type Usage ✅
**File:** `backend/inference.rs`

```rust
pub struct CandleInferenceBackend {
    pub(crate) model: Model,              // ✅ Direct enum
    pub(crate) tokenizer: Tokenizer,      // ✅ Direct type
    pub(crate) device: Device,            // ✅ Direct type
    model_size_bytes: u64,
}
```

**Assessment:** ✅ **CORRECT** - Uses Candle types directly
- No custom wrappers
- Direct access to model, tokenizer, device

### 3. Model Loading (Factory Pattern) ✅
**File:** `backend/models/mod.rs` lines 232-301

```rust
pub fn load_model(model_path: &str, device: &Device) -> Result<Model> {
    // Detect architecture
    let architecture = detect_architecture(&config_json)?;
    
    // Load appropriate model
    match architecture.as_str() {
        "llama" => {
            let model = llama::LlamaModel::load(path, device)?;
            Ok(Model::Llama(model))
        }
        "mistral" => {
            let model = mistral::MistralModel::load(path, device)?;
            Ok(Model::Mistral(model))
        }
        // ...
    }
}
```

**Assessment:** ✅ **GOOD** - Function-based factory
- Not a struct with methods
- Returns Model enum directly
- Candle-idiomatic pattern

### 4. Forward Pass Delegation ✅
**File:** `backend/models/mod.rs` lines 38-54

```rust
impl Model {
    pub fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        match self {
            Model::Llama(m) => m.forward(input_ids, position),
            Model::QuantizedLlama(m) => m.forward(input_ids, position),
            Model::Mistral(m) => m.forward(input_ids, position),
            Model::Phi(m) => m.forward(input_ids), // Phi doesn't use position
            // ...
        }
    }
}
```

**Assessment:** ✅ **PERFECT** - Delegates to natural interfaces
- Each model uses its own signature
- No forced abstraction
- Respects Candle's design

### 5. Inference Loop ✅
**File:** `backend/inference.rs` lines 225-270

```rust
for pos in 0..config.max_tokens {
    // Prepare input tensor
    let input_ids = Tensor::new(&tokens[..], &self.device)?
        .unsqueeze(0)?;
    
    // Forward pass (delegates to Model enum)
    let logits = self.model.forward(&input_ids, pos_usize)?;
    
    // Sample next token
    let next_token = logits_processor.sample(&logits)?;
    
    tokens.push(next_token);
}
```

**Assessment:** ✅ **CORRECT** - Direct Candle usage
- Direct tensor operations
- No unnecessary abstractions
- Clear, readable code

---

## ⚠️ Minor Deviations (Not Critical)

### 1. CandleInferenceBackend Struct ⚠️

**Current:**
```rust
pub struct CandleInferenceBackend {
    pub(crate) model: Model,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) device: Device,
    model_size_bytes: u64,
}
```

**Pure Candle would be:**
```rust
// Just use the types directly, no wrapper struct
let model = Model::Llama(...);
let tokenizer = Tokenizer::from_file(...)?;
let device = Device::Cpu;
```

**Assessment:** ⚠️ **ACCEPTABLE** - Minor wrapper for convenience
- Groups related components
- Still exposes Candle types directly (pub(crate))
- Not a heavy abstraction
- **Justified by repo idioms** (needs to impl InferenceBackend trait)

### 2. InferenceBackend Trait ⚠️

**File:** `http/mod.rs` (referenced in inference.rs)

```rust
#[async_trait]
impl InferenceBackend for CandleInferenceBackend {
    async fn execute(&mut self, prompt: &str, config: &SamplingConfig) -> Result<...>
}
```

**Assessment:** ⚠️ **ACCEPTABLE** - Needed for repo architecture
- Required for HTTP layer abstraction
- Doesn't hide Candle types
- Thin wrapper around Candle operations
- **Justified by repo idioms** (HTTP server needs trait object)

---

## 📊 Comparison: LLM Worker vs Old SD Worker

| Aspect | LLM Worker | Old SD Worker | Winner |
|--------|-----------|---------------|--------|
| Model types | ✅ Direct enum | ❌ Wrapped in structs | LLM |
| Tokenizer | ✅ Direct Tokenizer | ❌ Wrapped in ClipTextEncoder | LLM |
| VAE/Decoder | N/A | ❌ Wrapped in VaeDecoder | LLM |
| Pipeline | ✅ No pipeline struct | ❌ InferencePipeline struct | LLM |
| Forward pass | ✅ Delegates to natural interfaces | ❌ Forced through wrapper | LLM |
| Pattern | ✅ Enum-based | ❌ Struct-based | LLM |
| Abstraction | ✅ Minimal | ❌ Heavy | LLM |

**Score:** LLM Worker: 8/10, Old SD Worker: 3/10

---

## 🎯 Why LLM Worker is Better

### 1. Enum Pattern (Candle Idiom)
LLM worker uses `Model` enum, which is the **Candle-recommended pattern** for polymorphism.

**From TEAM-017 comments:**
> "switched to enum pattern for Candle idiomaticity"

### 2. No Unnecessary Wrappers
LLM worker doesn't wrap Candle types in custom structs like:
- ❌ ClipTextEncoder (SD worker had this)
- ❌ VaeDecoder (SD worker had this)
- ❌ InferencePipeline (SD worker had this)

### 3. Direct Type Usage
LLM worker exposes Candle types:
```rust
pub(crate) model: Model,      // Direct access
pub(crate) tokenizer: Tokenizer,  // Direct access
pub(crate) device: Device,    // Direct access
```

### 4. Natural Interfaces
Each model uses its natural interface:
```rust
Model::Phi(m) => m.forward(input_ids),  // Phi doesn't use position
Model::Llama(m) => m.forward(input_ids, position),  // Llama does
```

No forced abstraction!

---

## 🤔 Why the Minor Deviations?

### Repo Idioms Require Them

**1. HTTP Server Needs Trait:**
```rust
#[async_trait]
trait InferenceBackend {
    async fn execute(...) -> Result<...>;
}
```

**Why:** HTTP layer needs to work with different backends (CPU/CUDA/Metal)

**2. GenerationEngine Needs Struct:**
```rust
pub struct GenerationEngine {
    backend: Arc<Mutex<CandleInferenceBackend>>,
    // ...
}
```

**Why:** spawn_blocking pattern requires shared ownership

**3. RequestQueue Pattern:**
```rust
let (queue, rx) = RequestQueue::new();
```

**Why:** Async job processing architecture

---

## ✅ Final Verdict

### LLM Worker: 8/10 Candle Idiomatic

**What's Good:**
- ✅ Model enum pattern (Candle idiom)
- ✅ Direct type usage (no wrappers)
- ✅ Factory function (not struct)
- ✅ Natural interfaces (delegates correctly)
- ✅ Direct tensor operations

**Minor Deviations:**
- ⚠️ CandleInferenceBackend struct (justified by repo idioms)
- ⚠️ InferenceBackend trait (justified by HTTP architecture)

**Overall:** **Excellent balance** between Candle idioms and repo idioms!

---

## 🎯 Comparison Summary

| Worker | Candle Idiom Score | Notes |
|--------|-------------------|-------|
| **LLM Worker** | **8/10** ✅ | Excellent! Minor deviations justified |
| **Old SD Worker** | **3/10** ❌ | Heavy wrappers, not idiomatic |
| **New SD Worker** | **9/10** ✅ | Direct functions, no wrappers! |

---

## 📝 Key Takeaway

**LLM worker is ALREADY Candle idiomatic!**

The new SD worker (after Rule Zero) is **even more idiomatic** because:
- ✅ Uses functions (not even a backend struct)
- ✅ Zero wrappers (not even CandleInferenceBackend)
- ✅ Pure Candle patterns

But LLM worker's approach is **also valid** and **justified** by the repo architecture needs.

---

**Both are good! LLM worker showed us the way.** 🎯
