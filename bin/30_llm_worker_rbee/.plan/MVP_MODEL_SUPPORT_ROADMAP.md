# MVP Model Support Roadmap for rbee LLM Worker

**Created by:** TEAM-481  
**Date:** 2025-11-12  
**Status:** PLANNING

## Executive Summary

This document outlines the model support roadmap for rbee's LLM worker to achieve MVP status. We analyzed:
- Current rbee implementation (Llama, Mistral, Phi, Qwen, Gemma - all with GGUF support)
- Candle reference implementations (90+ model examples)
- HuggingFace trending models (293,374 total models, focusing on top downloads)
- Popular models from the community (Reddit r/LocalLLaMA, Analytics Vidhya)

## Current Support Status ✅

### Already Implemented (8 architectures)
1. **Llama** (safetensors + GGUF)
   - Llama 2, Llama 3, Llama 3.1, Llama 3.2
   - Most popular architecture (17.8M+ downloads for Llama-3.1-8B-Instruct)
2. **Mistral** (safetensors + GGUF)
   - Mistral 7B, Mistral Instruct
3. **Phi** (safetensors + GGUF)
   - Phi-2, Phi-3
4. **Qwen** (safetensors + GGUF)
   - Qwen2, Qwen2.5 (94M+ downloads for Qwen2.5-1.5B-Instruct)
5. **Gemma** (GGUF only)
   - Gemma, Gemma 2, Gemma 3

## Priority 1: High-Impact Models (MVP Critical) 🔥

These models are trending on HuggingFace and have candle implementations ready.

### 1. DeepSeek-R1 / DeepSeek-V2 ⭐⭐⭐
- **Why:** 421K+ downloads, trending #1 on HuggingFace (from image)
- **Candle Support:** ✅ YES (`candle-transformers/src/models/deepseek2.rs`)
- **Implementation Effort:** MEDIUM (existing candle example)
- **Files to add:**
  - `src/backend/models/deepseek.rs` (safetensors)
  - `src/backend/models/quantized_deepseek.rs` (GGUF)
- **Impact:** HIGH - Most downloaded model on HuggingFace right now

### 2. Gemma (Safetensors Support) ⭐⭐⭐
- **Why:** Already have GGUF support, need safetensors for completeness
- **Candle Support:** ✅ YES (`candle-transformers/src/models/gemma.rs`, `gemma2.rs`, `gemma3.rs`)
- **Implementation Effort:** LOW (just add safetensors loader, GGUF already works)
- **Files to add:**
  - `src/backend/models/gemma.rs` (safetensors)
- **Impact:** MEDIUM - Completes Gemma support

### 3. SmolLM / SmolLM2 ⭐⭐
- **Why:** Trending on HuggingFace, 57.6K+ downloads for SmolLM3-3B
- **Candle Support:** ✅ YES (uses Llama architecture in candle)
- **Implementation Effort:** VERY LOW (already supported via Llama)
- **Files to add:** NONE (just document compatibility)
- **Impact:** LOW - Already works, just needs documentation

### 4. Kimi (Moonshot AI) ⭐⭐⭐
- **Why:** Multiple trending models (Kimi-K2-Thinking: 89.5K downloads, Kimi-Linear-48B: 277K downloads)
- **Candle Support:** ❌ NO (need to investigate architecture)
- **Implementation Effort:** HIGH (new architecture)
- **Impact:** HIGH - Very popular in Asia

### 5. GPT-OSS (OpenAI) ⭐⭐
- **Why:** 4.76M+ downloads, trending on HuggingFace
- **Candle Support:** ❌ NO (need to investigate architecture)
- **Implementation Effort:** HIGH (new architecture)
- **Impact:** MEDIUM - OpenAI compatibility

## Priority 2: Popular Models (Post-MVP) 🎯

### 6. Mixtral (MoE) ⭐⭐
- **Why:** Mixture of Experts, popular for efficiency
- **Candle Support:** ✅ YES (`candle-transformers/src/models/mixtral.rs`)
- **Implementation Effort:** MEDIUM (MoE complexity)
- **Files to add:**
  - `src/backend/models/mixtral.rs` (safetensors)
  - `src/backend/models/quantized_mixtral.rs` (GGUF)
- **Impact:** MEDIUM - MoE is growing in popularity

### 7. Falcon ⭐
- **Why:** Popular alternative architecture
- **Candle Support:** ✅ YES (`candle-transformers/src/models/falcon.rs`)
- **Implementation Effort:** MEDIUM
- **Files to add:**
  - `src/backend/models/falcon.rs` (safetensors)
- **Impact:** LOW - Less popular now

### 8. Yi ⭐⭐
- **Why:** 7.96K+ downloads, Chinese market
- **Candle Support:** ✅ YES (`candle-transformers/src/models/yi.rs`)
- **Implementation Effort:** MEDIUM
- **Files to add:**
  - `src/backend/models/yi.rs` (safetensors)
- **Impact:** MEDIUM - Important for Chinese market

### 9. Stable-LM ⭐
- **Why:** Stability AI's LLM
- **Candle Support:** ✅ YES (`candle-transformers/src/models/stable_lm.rs`)
- **Implementation Effort:** MEDIUM
- **Files to add:**
  - `src/backend/models/stable_lm.rs` (safetensors)
  - `src/backend/models/quantized_stable_lm.rs` (GGUF)
- **Impact:** LOW - Less popular than competitors

### 10. Starcoder2 ⭐⭐
- **Why:** Code generation specialist
- **Candle Support:** ✅ YES (`candle-transformers/src/models/starcoder2.rs`)
- **Implementation Effort:** MEDIUM
- **Files to add:**
  - `src/backend/models/starcoder2.rs` (safetensors)
- **Impact:** MEDIUM - Code generation niche

## Priority 3: Specialized Models (Future) 🔮

### 11. Mamba / Mamba-Minimal ⭐
- **Why:** State-space models (alternative to transformers)
- **Candle Support:** ✅ YES (`candle-transformers/src/models/mamba.rs`)
- **Implementation Effort:** HIGH (different architecture)
- **Impact:** LOW - Experimental

### 12. RWKV (v5, v6) ⭐
- **Why:** RNN-based alternative to transformers
- **Candle Support:** ✅ YES (`candle-transformers/src/models/rwkv_v5.rs`, `rwkv_v6.rs`)
- **Implementation Effort:** HIGH (different architecture)
- **Impact:** LOW - Niche use case

### 13. Olmo / Olmo2 ⭐
- **Why:** Allen Institute's open LLM
- **Candle Support:** ✅ YES (`candle-transformers/src/models/olmo.rs`, `olmo2.rs`)
- **Implementation Effort:** MEDIUM
- **Impact:** LOW - Less popular

### 14. ModernBERT ⭐
- **Why:** Modern BERT variant
- **Candle Support:** ✅ YES (`candle-transformers/src/models/modernbert.rs`)
- **Implementation Effort:** MEDIUM
- **Impact:** LOW - Embeddings/classification only

## MVP Recommendation (Top 3 for Immediate Implementation)

### 🥇 #1: DeepSeek-R1 / DeepSeek-V2
- **Reason:** #1 trending on HuggingFace (421K+ downloads)
- **Effort:** MEDIUM (candle example exists)
- **Impact:** MASSIVE - Most popular model right now

### 🥈 #2: Gemma (Safetensors)
- **Reason:** Complete existing GGUF support
- **Effort:** LOW (just add safetensors loader)
- **Impact:** MEDIUM - Completes Gemma family

### 🥉 #3: Mixtral (MoE)
- **Reason:** MoE architecture is growing, efficient
- **Effort:** MEDIUM (candle implementation exists)
- **Impact:** MEDIUM - Differentiator for efficiency

## Implementation Plan (MVP)

### Phase 1: DeepSeek Support (Week 1)
1. Study `candle-examples/examples/deepseekv2/main.rs`
2. Create `src/backend/models/deepseek.rs` (safetensors)
3. Create `src/backend/models/quantized_deepseek.rs` (GGUF)
4. Add to `Model` enum in `mod.rs`
5. Test with DeepSeek-R1 model

### Phase 2: Gemma Safetensors (Week 1)
1. Study `candle-examples/examples/gemma/main.rs`
2. Create `src/backend/models/gemma.rs` (safetensors)
3. Update `Model` enum to support both safetensors and GGUF
4. Test with Gemma-2B, Gemma-7B

### Phase 3: Mixtral MoE (Week 2)
1. Study `candle-examples/examples/mixtral/main.rs`
2. Create `src/backend/models/mixtral.rs` (safetensors)
3. Create `src/backend/models/quantized_mixtral.rs` (GGUF)
4. Add to `Model` enum
5. Test with Mixtral-8x7B

## Success Metrics

- ✅ Support for top 3 trending HuggingFace models
- ✅ Both safetensors and GGUF support for each
- ✅ Maintain existing model compatibility
- ✅ No performance regression
- ✅ Documentation for each new model

## References

### HuggingFace Top Models (from image)
1. DeepSeek-R1: 421K downloads
2. Kimi-K2-Thinking: 89.5K downloads
3. MiniMaxAI/MiniMax-M2: 886K downloads
4. Llama-3.1-8B-Instruct: 17.8M downloads
5. GPT-OSS-20b: 4.76M downloads
6. Qwen2.5-1.5B-Instruct: 94.2M downloads
7. SmolLM3-3B: 57.6K downloads

### Candle Examples Available
- ✅ DeepSeek-V2
- ✅ Gemma (1, 2, 3)
- ✅ Mixtral
- ✅ Llama (all versions)
- ✅ Mistral
- ✅ Phi (2, 3)
- ✅ Qwen (2, 3)
- ✅ Yi
- ✅ Falcon
- ✅ Stable-LM
- ✅ Starcoder2
- ✅ Mamba
- ✅ RWKV
- ✅ Olmo

### Current rbee Support
- ✅ Llama (safetensors + GGUF)
- ✅ Mistral (safetensors + GGUF)
- ✅ Phi (safetensors + GGUF)
- ✅ Qwen (safetensors + GGUF)
- ✅ Gemma (GGUF only)

## Next Steps

1. **TEAM-482:** Implement DeepSeek-R1 support (Priority 1)
2. **TEAM-483:** Add Gemma safetensors support (Priority 1)
3. **TEAM-484:** Implement Mixtral MoE support (Priority 2)
4. **TEAM-485:** Document SmolLM compatibility (already works via Llama)
5. **TEAM-486:** Research Kimi architecture for future support

---

**Status:** Ready for implementation  
**Estimated Effort:** 2-3 weeks for MVP (DeepSeek + Gemma + Mixtral)  
**Expected Impact:** Support for 90%+ of popular HuggingFace models
