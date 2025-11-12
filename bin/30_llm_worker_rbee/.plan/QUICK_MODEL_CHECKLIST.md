# Quick Model Support Checklist

**Created by:** TEAM-481  
**Date:** 2025-11-12

## MVP Models (Implement These First)

### ✅ Already Supported (8 architectures)
- [x] Llama (safetensors + GGUF) - 17.8M+ downloads
- [x] Mistral (safetensors + GGUF)
- [x] Phi (safetensors + GGUF)
- [x] Qwen (safetensors + GGUF) - 94.2M+ downloads
- [x] Gemma (GGUF only) - needs safetensors

### 🔥 Priority 1: MVP Critical (Implement Next)

#### 1. DeepSeek-R1 / DeepSeek-V2 ⭐⭐⭐
- [ ] Study candle example: `candle-examples/examples/deepseekv2/main.rs`
- [ ] Create `src/backend/models/deepseek.rs` (safetensors)
- [ ] Create `src/backend/models/quantized_deepseek.rs` (GGUF)
- [ ] Add to `Model` enum in `mod.rs`
- [ ] Add to `detect_architecture()` function
- [ ] Add to `load_model()` function
- [ ] Test with DeepSeek-R1 GGUF
- [ ] Test with DeepSeek-V2 safetensors
- **Downloads:** 421K+ (trending #1)
- **Candle Support:** ✅ YES
- **Effort:** MEDIUM

#### 2. Gemma (Safetensors Support) ⭐⭐⭐
- [ ] Study candle example: `candle-examples/examples/gemma/main.rs`
- [ ] Create `src/backend/models/gemma.rs` (safetensors)
- [ ] Update `Model` enum to support both Gemma variants
- [ ] Add to `detect_architecture()` function
- [ ] Add to `load_model()` function
- [ ] Test with Gemma-2B safetensors
- [ ] Test with Gemma-7B safetensors
- [ ] Test with Gemma-2 safetensors
- **Downloads:** N/A (completing existing support)
- **Candle Support:** ✅ YES
- **Effort:** LOW

#### 3. Mixtral (MoE) ⭐⭐
- [ ] Study candle example: `candle-examples/examples/mixtral/main.rs`
- [ ] Create `src/backend/models/mixtral.rs` (safetensors)
- [ ] Create `src/backend/models/quantized_mixtral.rs` (GGUF)
- [ ] Add to `Model` enum in `mod.rs`
- [ ] Add to `detect_architecture()` function
- [ ] Add to `load_model()` function
- [ ] Test with Mixtral-8x7B GGUF
- [ ] Test with Mixtral-8x7B safetensors
- **Downloads:** Popular MoE model
- **Candle Support:** ✅ YES
- **Effort:** MEDIUM

### 🎯 Priority 2: Post-MVP (Implement Later)

#### 4. Yi ⭐⭐
- [ ] Study candle example: `candle-examples/examples/yi/main.rs`
- [ ] Create `src/backend/models/yi.rs` (safetensors)
- [ ] Add to `Model` enum
- **Downloads:** 7.96K+
- **Candle Support:** ✅ YES
- **Effort:** MEDIUM

#### 5. Starcoder2 ⭐⭐
- [ ] Study candle example: `candle-examples/examples/starcoder2/main.rs`
- [ ] Create `src/backend/models/starcoder2.rs` (safetensors)
- [ ] Add to `Model` enum
- **Downloads:** Code generation specialist
- **Candle Support:** ✅ YES
- **Effort:** MEDIUM

#### 6. Falcon ⭐
- [ ] Study candle example: `candle-examples/examples/falcon/main.rs`
- [ ] Create `src/backend/models/falcon.rs` (safetensors)
- [ ] Add to `Model` enum
- **Downloads:** Moderate
- **Candle Support:** ✅ YES
- **Effort:** MEDIUM

#### 7. Stable-LM ⭐
- [ ] Study candle example: `candle-examples/examples/stable-lm/main.rs`
- [ ] Create `src/backend/models/stable_lm.rs` (safetensors)
- [ ] Create `src/backend/models/quantized_stable_lm.rs` (GGUF)
- [ ] Add to `Model` enum
- **Downloads:** Moderate
- **Candle Support:** ✅ YES
- **Effort:** MEDIUM

### 🔮 Priority 3: Future/Experimental

#### 8. Mamba ⭐
- [ ] Study candle example: `candle-examples/examples/mamba/main.rs`
- [ ] Create `src/backend/models/mamba.rs`
- [ ] Add to `Model` enum
- **Downloads:** Experimental
- **Candle Support:** ✅ YES
- **Effort:** HIGH (different architecture)

#### 9. RWKV (v5, v6) ⭐
- [ ] Study candle example: `candle-examples/examples/rwkv/main.rs`
- [ ] Create `src/backend/models/rwkv.rs`
- [ ] Add to `Model` enum
- **Downloads:** Niche
- **Candle Support:** ✅ YES
- **Effort:** HIGH (different architecture)

#### 10. Olmo / Olmo2 ⭐
- [ ] Study candle example: `candle-examples/examples/olmo/main.rs`
- [ ] Create `src/backend/models/olmo.rs`
- [ ] Add to `Model` enum
- **Downloads:** Moderate
- **Candle Support:** ✅ YES
- **Effort:** MEDIUM

## Models Requiring Research (No Candle Support Yet)

### Kimi (Moonshot AI) ⭐⭐⭐
- [ ] Research architecture (likely Llama-based?)
- [ ] Check if compatible with existing Llama loader
- [ ] If not, implement custom loader
- **Downloads:** 89.5K+ (Kimi-K2-Thinking), 277K+ (Kimi-Linear-48B)
- **Candle Support:** ❌ UNKNOWN
- **Effort:** HIGH (needs research)

### GPT-OSS (OpenAI) ⭐⭐
- [ ] Research architecture
- [ ] Check if compatible with existing loaders
- [ ] If not, implement custom loader
- **Downloads:** 4.76M+
- **Candle Support:** ❌ UNKNOWN
- **Effort:** HIGH (needs research)

### MiniMaxAI/MiniMax-M2 ⭐⭐
- [ ] Research architecture
- [ ] Check if compatible with existing loaders
- [ ] If not, implement custom loader
- **Downloads:** 886K+
- **Candle Support:** ❌ UNKNOWN
- **Effort:** HIGH (needs research)

## Already Compatible (Just Document)

### SmolLM / SmolLM2 ✅
- **Status:** Already works via Llama architecture
- **Action:** Just add documentation
- **Downloads:** 57.6K+ (SmolLM3-3B)
- **Candle Support:** ✅ YES (uses Llama)
- **Effort:** NONE (documentation only)

## Implementation Order (Recommended)

1. **Week 1, Day 1-3:** DeepSeek-R1 (trending #1, high impact)
2. **Week 1, Day 4-5:** Gemma safetensors (low effort, completes existing support)
3. **Week 2, Day 1-5:** Mixtral MoE (medium effort, MoE differentiator)
4. **Week 3:** Yi, Starcoder2, or Falcon (based on user demand)

## Success Criteria

- [ ] All Priority 1 models working (DeepSeek, Gemma, Mixtral)
- [ ] Both safetensors and GGUF support for each
- [ ] No regression in existing models
- [ ] Documentation updated
- [ ] Integration tests passing

## Files to Modify for Each New Model

### For Safetensors Support:
1. Create `src/backend/models/{model_name}.rs`
2. Update `src/backend/models/mod.rs`:
   - Add module declaration
   - Add variant to `Model` enum
   - Add to `forward()`, `eos_token_id()`, `architecture()`, `vocab_size()`, `reset_cache()`
   - Add to `detect_architecture()` function
   - Add to `load_model()` function

### For GGUF Support:
1. Create `src/backend/models/quantized_{model_name}.rs`
2. Update `src/backend/models/mod.rs`:
   - Add module declaration
   - Add variant to `Model` enum
   - Add to all match statements
   - Add to `detect_architecture_from_gguf()` function
   - Add to GGUF loading in `load_model()` function

## Testing Checklist (Per Model)

- [ ] Download model from HuggingFace
- [ ] Test safetensors loading
- [ ] Test GGUF loading
- [ ] Test inference (generate text)
- [ ] Test streaming
- [ ] Test cache reset
- [ ] Test with different quantization levels (Q4, Q5, Q8)
- [ ] Verify EOS token handling
- [ ] Check memory usage
- [ ] Benchmark performance

---

**Next Action:** Start with DeepSeek-R1 implementation (Priority 1, #1)
