# Model Support Summary for rbee LLM Worker

**Created by:** TEAM-481  
**Date:** 2025-11-12  
**Status:** PLANNING COMPLETE

## TL;DR - What to Implement for MVP

### Top 3 Models for Immediate Implementation:

1. **🥇 DeepSeek-R1 / DeepSeek-V2** - Trending #1 on HuggingFace (421K+ downloads)
2. **🥈 Gemma (Safetensors)** - Complete existing GGUF support
3. **🥉 Mixtral (MoE)** - Mixture of Experts, efficient and popular

**Estimated Effort:** 2-3 weeks  
**Expected Impact:** Support for 90%+ of popular HuggingFace models

---

## Current Status

### ✅ Already Supported (8 architectures)
| Model | Safetensors | GGUF | Downloads | Status |
|-------|-------------|------|-----------|--------|
| Llama | ✅ | ✅ | 17.8M+ | Complete |
| Mistral | ✅ | ✅ | High | Complete |
| Phi | ✅ | ✅ | High | Complete |
| Qwen | ✅ | ✅ | 94.2M+ | Complete |
| Gemma | ❌ | ✅ | High | GGUF only |

**Total:** 5 model families, 8 architectures

---

## MVP Additions (Priority 1)

### 1. DeepSeek-R1 / DeepSeek-V2 ⭐⭐⭐
- **Why:** #1 trending on HuggingFace right now
- **Downloads:** 421K+ (DeepSeek-R1)
- **Candle Support:** ✅ YES (`candle-examples/examples/deepseekv2/`)
- **Effort:** MEDIUM (2-3 days)
- **Files to create:**
  - `src/backend/models/deepseek.rs` (safetensors)
  - `src/backend/models/quantized_deepseek.rs` (GGUF)

### 2. Gemma (Safetensors) ⭐⭐⭐
- **Why:** Complete existing GGUF support
- **Downloads:** High (Google's model)
- **Candle Support:** ✅ YES (`candle-examples/examples/gemma/`)
- **Effort:** LOW (1-2 days)
- **Files to create:**
  - `src/backend/models/gemma.rs` (safetensors)

### 3. Mixtral (MoE) ⭐⭐
- **Why:** Mixture of Experts, efficient architecture
- **Downloads:** High (Mistral AI)
- **Candle Support:** ✅ YES (`candle-examples/examples/mixtral/`)
- **Effort:** MEDIUM (2-3 days)
- **Files to create:**
  - `src/backend/models/mixtral.rs` (safetensors)
  - `src/backend/models/quantized_mixtral.rs` (GGUF)

---

## Post-MVP Additions (Priority 2)

### 4. Yi ⭐⭐
- **Downloads:** 7.96K+
- **Candle Support:** ✅ YES
- **Effort:** MEDIUM (2-3 days)

### 5. Starcoder2 ⭐⭐
- **Use Case:** Code generation specialist
- **Candle Support:** ✅ YES
- **Effort:** MEDIUM (2-3 days)

### 6. Falcon ⭐
- **Downloads:** Moderate
- **Candle Support:** ✅ YES
- **Effort:** MEDIUM (2-3 days)

### 7. Stable-LM ⭐
- **Downloads:** Moderate (Stability AI)
- **Candle Support:** ✅ YES
- **Effort:** MEDIUM (2-3 days)

---

## Future/Experimental (Priority 3)

### 8. Mamba ⭐
- **Architecture:** State-space models (alternative to transformers)
- **Candle Support:** ✅ YES
- **Effort:** HIGH (different architecture)

### 9. RWKV (v5, v6) ⭐
- **Architecture:** RNN-based alternative
- **Candle Support:** ✅ YES
- **Effort:** HIGH (different architecture)

### 10. Olmo / Olmo2 ⭐
- **Downloads:** Moderate (Allen Institute)
- **Candle Support:** ✅ YES
- **Effort:** MEDIUM (2-3 days)

---

## Models Requiring Research (No Candle Support Yet)

### Kimi (Moonshot AI) ⭐⭐⭐
- **Downloads:** 89.5K+ (Kimi-K2-Thinking), 277K+ (Kimi-Linear-48B)
- **Candle Support:** ❌ UNKNOWN (needs research)
- **Effort:** HIGH (architecture unknown)

### GPT-OSS (OpenAI) ⭐⭐
- **Downloads:** 4.76M+
- **Candle Support:** ❌ UNKNOWN (needs research)
- **Effort:** HIGH (architecture unknown)

### MiniMaxAI/MiniMax-M2 ⭐⭐
- **Downloads:** 886K+
- **Candle Support:** ❌ UNKNOWN (needs research)
- **Effort:** HIGH (architecture unknown)

---

## Already Compatible (Just Document)

### SmolLM / SmolLM2 ✅
- **Status:** Already works via Llama architecture
- **Downloads:** 57.6K+ (SmolLM3-3B)
- **Action:** Add documentation only
- **Effort:** NONE

---

## Implementation Timeline

### Week 1: DeepSeek + Gemma
- **Day 1-3:** DeepSeek-R1 implementation
  - Study candle example
  - Create safetensors loader
  - Create GGUF loader
  - Add to model enum
  - Test with DeepSeek-R1
- **Day 4-5:** Gemma safetensors
  - Study candle example
  - Create safetensors loader
  - Test with Gemma-2B, Gemma-7B

### Week 2: Mixtral
- **Day 1-5:** Mixtral MoE implementation
  - Study candle example
  - Create safetensors loader
  - Create GGUF loader
  - Add to model enum
  - Test with Mixtral-8x7B

### Week 3+: Post-MVP
- Yi, Starcoder2, Falcon, Stable-LM (based on user demand)

---

## Success Metrics

- ✅ Support for top 3 trending HuggingFace models
- ✅ Both safetensors and GGUF support for each
- ✅ Maintain existing model compatibility
- ✅ No performance regression
- ✅ Documentation for each new model
- ✅ Integration tests passing

---

## Key Insights from Research

### HuggingFace Trending Models (from screenshot)
1. **DeepSeek-R1:** 421K downloads (trending #1)
2. **Kimi-K2-Thinking:** 89.5K downloads
3. **MiniMaxAI/MiniMax-M2:** 886K downloads
4. **Llama-3.1-8B-Instruct:** 17.8M downloads ✅ (already supported)
5. **GPT-OSS-20b:** 4.76M downloads
6. **Qwen2.5-1.5B-Instruct:** 94.2M downloads ✅ (already supported)
7. **SmolLM3-3B:** 57.6K downloads ✅ (already compatible via Llama)

### Candle Support Status
- **90+ model examples** in candle-examples
- **100+ model implementations** in candle-transformers
- **Strong support for:** Llama, Mistral, Phi, Qwen, Gemma, DeepSeek, Mixtral, Yi, Falcon, Stable-LM, Starcoder2

### rbee Current Support
- **5 model families:** Llama, Mistral, Phi, Qwen, Gemma
- **8 architectures:** Including quantized versions
- **Both formats:** Safetensors and GGUF (except Gemma safetensors)

---

## Recommendations

### Immediate Actions (This Week)
1. ✅ **Approve this plan**
2. 🔥 **Start DeepSeek-R1 implementation** (highest priority)
3. 🔥 **Complete Gemma safetensors support** (low effort, high value)

### Next Week
1. 🎯 **Implement Mixtral MoE** (differentiator)
2. 📝 **Document SmolLM compatibility** (already works)

### Future Work
1. 🔮 **Research Kimi architecture** (high downloads, unknown support)
2. 🔮 **Research GPT-OSS architecture** (high downloads, unknown support)
3. 🎯 **Implement Yi, Starcoder2, Falcon** (based on user demand)

---

## Files Created

1. **MVP_MODEL_SUPPORT_ROADMAP.md** - Comprehensive analysis and roadmap
2. **QUICK_MODEL_CHECKLIST.md** - Implementation checklist with tasks
3. **MODEL_SUPPORT_SUMMARY.md** - This summary document

---

## Next Steps

**TEAM-482:** Implement DeepSeek-R1 support (Priority 1)  
**TEAM-483:** Add Gemma safetensors support (Priority 1)  
**TEAM-484:** Implement Mixtral MoE support (Priority 2)

---

**Status:** ✅ PLANNING COMPLETE - Ready for implementation  
**Estimated Effort:** 2-3 weeks for MVP  
**Expected Impact:** Support for 90%+ of popular HuggingFace models
