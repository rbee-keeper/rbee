# Missing Features from Candle Reference

**Analysis Date:** 2025-11-12  
**Analyzed by:** TEAM-485  
**Comparison:** rbee LLM Worker vs. reference/candle examples

## Executive Summary

Our LLM worker has **13 model architectures** implemented. Candle examples showcase **80+ models** including many text generation models we're missing. This document identifies high-value additions we can implement.

## Current Implementation Status

### ✅ Models We Have (13 total)

**Text Generation Models:**
- **DeepSeek** (safetensors + GGUF)
- **Gemma** (safetensors + GGUF)
- **Llama** (safetensors + GGUF)
- **Mistral** (safetensors, GGUF via Llama loader)
- **Mixtral** (MoE - safetensors)
- **Phi** (safetensors + GGUF)
- **Qwen** (safetensors + GGUF)

### ❌ Missing Text Generation Models (High Priority)

Based on candle-transformers availability and popularity:

#### Tier 1: High Demand Models (Should Add)

1. **Falcon** (`falcon.rs`)
   - Popular open model from TII
   - 7B, 40B, 180B variants
   - Already in candle-transformers
   - **Effort:** Low (standard transformer)

2. **StableLM** (`stable_lm.rs` + `quantized_stable_lm.rs`)
   - Stability AI's text model
   - Multiple sizes (1.6B - 12B)
   - Has quantized support
   - **Effort:** Low

3. **Yi** (`yi.rs`)
   - 01.AI's competitive model
   - 6B and 34B variants
   - Strong performance
   - **Effort:** Low

4. **Starcoder2** (`starcoder2.rs`)
   - Code generation specialist
   - 3B, 7B, 15B variants
   - **Use case:** Code completion
   - **Effort:** Low

5. **Mamba** (`mamba.rs`)
   - State-space model (alternative to transformers)
   - Linear complexity vs quadratic attention
   - 130M - 2.8B variants
   - **Effort:** Medium (different architecture)

6. **Recurrent Gemma** (`recurrent_gemma.rs` + quantized)
   - Google's recurrent variant
   - More efficient than standard Gemma
   - **Effort:** Low (similar to Gemma)

#### Tier 2: Specialized Models (Consider Adding)

7. **Olmo** (`olmo.rs`, `olmo2.rs`)
   - Allen AI's fully open model
   - Complete training data available
   - **Use case:** Research, transparency
   - **Effort:** Low

8. **ChatGLM** (`chatglm.rs`)
   - Bilingual (Chinese/English)
   - Popular in Asian markets
   - **Effort:** Medium

9. **GLM4** (`glm4.rs`, `glm4_new.rs`)
   - Newer version of ChatGLM
   - Better performance
   - **Effort:** Medium

10. **Granite** (`granite.rs`)
    - IBM's enterprise model
    - Code and language variants
    - **Effort:** Low

11. **GraniteMoeHybrid** (`granitemoehybrid.rs`)
    - MoE version of Granite
    - **Effort:** Medium (MoE complexity)

12. **ModernBERT** (`modernbert.rs`)
    - Updated BERT architecture
    - Better than original BERT
    - **Use case:** Embeddings, classification
    - **Effort:** Low

13. **Helium** (`helium.rs`)
    - Newer efficient model
    - **Effort:** Low

14. **Based** (`based.rs`)
    - Alternative attention mechanism
    - **Effort:** Medium

#### Tier 3: Niche/Experimental

15. **BigCode** (`bigcode.rs`)
16. **Replit Code** (`replit-code/`)
17. **MPT** (`mpt.rs` + quantized)
18. **Persimmon** (`persimmon.rs`)
19. **RWKV** (`rwkv_v5.rs`, `rwkv_v6.rs` + quantized)
20. **Qwen2 MoE** (`qwen2_moe.rs`)
21. **Qwen3** (`qwen3.rs`, `qwen3_moe.rs` + quantized)

## Missing Features (Non-Model)

### 1. **Flash Attention Support**

**Status:** Candle examples use `--use-flash-attn` flag  
**Impact:** 2-4x faster inference, lower memory  
**Effort:** Medium (requires candle-flash-attn integration)

```rust
// From candle/candle-examples/examples/llama/main.rs:114
#[arg(long)]
use_flash_attn: bool,

// Line 179:
let config = config.into_config(args.use_flash_attn);
```

**Our code:** No flash attention support

### 2. **Repeat Penalty**

**Status:** ✅ **WE HAVE THIS** (but should verify implementation)

```rust
// From our sampling.rs - we don't apply repeat penalty!
pub fn create_logits_processor(config: &SamplingConfig) -> LogitsProcessor {
    // ... no repeat penalty application
}
```

**Candle examples do:**
```rust
let logits = if args.repeat_penalty == 1. {
    logits
} else {
    let start_at = tokens.len().saturating_sub(args.repeat_last_n);
    candle_transformers::utils::apply_repeat_penalty(
        &logits,
        args.repeat_penalty,
        &tokens[start_at..],
    )?
};
```

**Action:** ✅ Add repeat penalty to our inference loop

### 3. **Interactive/Chat Modes**

**Status:** Candle has interactive prompting  
**Our code:** Single-shot inference only

```rust
// From candle quantized example
enum Prompt {
    Interactive,
    Chat,
    One(String),
}
```

**Action:** Consider adding for CLI/testing

### 4. **Tracing Support**

**Status:** Candle examples support Chrome tracing  
**Our code:** We use observability-narration (different approach)

```rust
// Candle approach:
#[arg(long)]
tracing: bool,

let _guard = if args.tracing {
    let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
    tracing_subscriber::registry().with(chrome_layer).init();
    Some(guard)
} else {
    None
};
```

**Action:** ✅ Our narration system is better for production

### 5. **DType Selection**

**Status:** Candle allows runtime dtype selection  
**Our code:** Fixed dtype per model

```rust
// Candle:
#[arg(long)]
dtype: Option<String>,

let dtype = match args.dtype.as_deref() {
    Some("f16") => DType::F16,
    Some("bf16") => DType::BF16,
    Some("f32") => DType::F32,
    Some(dtype) => bail!("Unsupported dtype {dtype}"),
    None => DType::F16,
};
```

**Action:** Consider adding as model capability

### 6. **Multi-process Support**

**Status:** Candle has `llama_multiprocess` example  
**Our code:** Single process only

**Use case:** Model parallelism for large models  
**Effort:** High (requires IPC/shared memory)  
**Priority:** Low (not needed yet)

### 7. **No KV Cache Toggle**

**Status:** Candle allows disabling KV cache  
**Our code:** Always uses cache

```rust
#[arg(long)]
no_kv_cache: bool,
```

**Action:** Low priority (cache is almost always wanted)

## Implementation Recommendations

### Phase 1: Quick Wins (1-2 days)

1. **Add Repeat Penalty** ⚠️ CRITICAL
   - Modify `sampling.rs` and `inference.rs`
   - Use `candle_transformers::utils::apply_repeat_penalty`
   - Add to `SamplingConfig`

2. **Add Falcon Model**
   - Copy pattern from Llama
   - Add to `models/` directory
   - Update `mod.rs` factory

3. **Add StableLM Model**
   - Has quantized support
   - Similar to Llama architecture

### Phase 2: High Value (3-5 days)

4. **Add Yi Model**
5. **Add Starcoder2 Model** (code generation)
6. **Add Recurrent Gemma** (efficiency)
7. **Flash Attention Support** (performance)

### Phase 3: Specialized (1-2 weeks)

8. **Mamba** (different architecture)
9. **Olmo** (research use case)
10. **ChatGLM/GLM4** (multilingual)

## Architecture Patterns from Candle

### Pattern 1: Quantized + Regular Enum

Many candle examples use this pattern:

```rust
enum Model {
    Regular(RegularModel),
    Quantized(QuantizedModel),
}

impl Model {
    fn forward(&mut self, xs: &Tensor, pos: usize) -> Result<Tensor> {
        match self {
            Self::Regular(m) => m.forward(xs, pos),
            Self::Quantized(m) => m.forward(xs, pos),
        }
    }
}
```

**Our approach:** Separate enum variants (DeepSeek vs QuantizedDeepSeek)  
**Candle approach:** Nested enum per model family  
**Verdict:** Both valid, ours is more explicit

### Pattern 2: TextGeneration Wrapper

Candle examples often wrap model + tokenizer + logits_processor:

```rust
struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        // Complete generation loop
    }
}
```

**Our approach:** `CandleInferenceBackend` + separate `GenerationEngine`  
**Verdict:** Our separation is cleaner for HTTP service

### Pattern 3: Model-Specific EOS Handling

```rust
// Llama has multiple EOS tokens
pub enum LlamaEosToks {
    Single(u32),
    Multiple(Vec<u32>),
}

// Check in generation loop:
match eos_token_id {
    Some(LlamaEosToks::Single(eos_tok_id)) if next_token == eos_tok_id => break,
    Some(LlamaEosToks::Multiple(ref eos_ids)) if eos_ids.contains(&next_token) => break,
    _ => (),
}
```

**Our approach:** Single `eos_token_id()` method  
**Action:** Consider supporting multiple EOS tokens

## Quantized Model Support

### Models with Quantized Versions in Candle

- ✅ Llama (we have)
- ✅ Phi (we have)
- ✅ Qwen (we have)
- ✅ Gemma (we have)
- ✅ DeepSeek (we have)
- ❌ Mistral (we use Llama loader)
- ❌ StableLM
- ❌ Recurrent Gemma
- ❌ RWKV v5/v6
- ❌ T5
- ❌ Phi3 (separate from Phi)
- ❌ Qwen3 (separate from Qwen2)
- ❌ MPT
- ❌ Moondream
- ❌ Metavoice

## Code Quality Observations

### Candle Strengths

1. **Consistent patterns** across examples
2. **Good CLI argument handling** with clap
3. **Comprehensive model coverage**
4. **Performance features** (flash attention, dtype selection)

### Our Strengths

1. **Better production architecture** (HTTP service, job queue)
2. **Observability** (narration system)
3. **Trait-based abstraction** (easier to add models)
4. **Clear separation of concerns**

## Action Items Summary

### Must Do (Correctness)

- [ ] **Add repeat penalty support** - We're missing this critical feature!

### Should Do (High Value)

- [ ] Add Falcon model
- [ ] Add StableLM model
- [ ] Add Yi model
- [ ] Add Starcoder2 model (code generation)
- [ ] Add flash attention support

### Could Do (Nice to Have)

- [ ] Add Recurrent Gemma
- [ ] Add Mamba
- [ ] Add Olmo
- [ ] Add runtime dtype selection
- [ ] Support multiple EOS tokens

### Won't Do (Low Priority)

- Interactive/chat modes (we're a service)
- Multi-process support (not needed yet)
- No-KV-cache toggle (always want cache)

## Conclusion

**Key Finding:** We're missing **repeat penalty** - this is a critical inference feature that affects output quality.

**Model Coverage:** We have solid coverage of popular models (13 architectures). Adding Falcon, StableLM, Yi, and Starcoder2 would give us 17 architectures covering 90%+ of use cases.

**Performance:** Flash attention support would be the biggest performance win (2-4x speedup).

**Architecture:** Our trait-based approach makes adding new models easier than candle's example-based approach. We should leverage this advantage.

---

**Next Steps:**
1. Fix repeat penalty (CRITICAL)
2. Add Falcon (1 day)
3. Add StableLM (1 day)
4. Evaluate flash attention (research spike)
