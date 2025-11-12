# Model Support Matrix for rbee LLM Worker

**Created by:** TEAM-481  
**Date:** 2025-11-12

## Legend
- ✅ = Fully supported
- 🟡 = Partially supported (one format only)
- ❌ = Not supported
- 🔍 = Needs research
- 📝 = Documentation only (already compatible)

---

## Current Support Matrix

| Model | Safetensors | GGUF | Candle Support | HF Downloads | Priority | Status |
|-------|-------------|------|----------------|--------------|----------|--------|
| **Llama** | ✅ | ✅ | ✅ | 17.8M+ | ✅ MVP | Complete |
| **Mistral** | ✅ | ✅ | ✅ | High | ✅ MVP | Complete |
| **Phi** | ✅ | ✅ | ✅ | High | ✅ MVP | Complete |
| **Qwen** | ✅ | ✅ | ✅ | 94.2M+ | ✅ MVP | Complete |
| **Gemma** | ❌ | ✅ | ✅ | High | 🔥 P1 | GGUF only |
| **DeepSeek** | ❌ | ❌ | ✅ | 421K+ | 🔥 P1 | Not implemented |
| **Mixtral** | ❌ | ❌ | ✅ | High | 🔥 P1 | Not implemented |
| **SmolLM** | 📝 | 📝 | ✅ (Llama) | 57.6K+ | 📝 Doc | Already compatible |
| **Yi** | ❌ | ❌ | ✅ | 7.96K+ | 🎯 P2 | Not implemented |
| **Starcoder2** | ❌ | ❌ | ✅ | Moderate | 🎯 P2 | Not implemented |
| **Falcon** | ❌ | ❌ | ✅ | Moderate | 🎯 P2 | Not implemented |
| **Stable-LM** | ❌ | ❌ | ✅ | Moderate | 🎯 P2 | Not implemented |
| **Mamba** | ❌ | ❌ | ✅ | Low | 🔮 P3 | Not implemented |
| **RWKV** | ❌ | ❌ | ✅ | Low | 🔮 P3 | Not implemented |
| **Olmo** | ❌ | ❌ | ✅ | Moderate | 🔮 P3 | Not implemented |
| **Kimi** | ❌ | ❌ | 🔍 | 277K+ | 🔍 Research | Unknown architecture |
| **GPT-OSS** | ❌ | ❌ | 🔍 | 4.76M+ | 🔍 Research | Unknown architecture |
| **MiniMax-M2** | ❌ | ❌ | 🔍 | 886K+ | 🔍 Research | Unknown architecture |

---

## Detailed Model Information

### ✅ Fully Supported (5 families, 8 architectures)

#### 1. Llama Family
- **Variants:** Llama 2, Llama 3, Llama 3.1, Llama 3.2
- **Formats:** Safetensors ✅, GGUF ✅
- **Downloads:** 17.8M+ (Llama-3.1-8B-Instruct)
- **Files:**
  - `src/backend/models/llama.rs` (safetensors)
  - `src/backend/models/quantized_llama.rs` (GGUF)
- **Status:** ✅ Complete

#### 2. Mistral Family
- **Variants:** Mistral 7B, Mistral Instruct
- **Formats:** Safetensors ✅, GGUF ✅
- **Downloads:** High
- **Files:**
  - `src/backend/models/mistral.rs` (safetensors)
  - Uses `quantized_llama.rs` for GGUF (same format)
- **Status:** ✅ Complete

#### 3. Phi Family
- **Variants:** Phi-2, Phi-3
- **Formats:** Safetensors ✅, GGUF ✅
- **Downloads:** High
- **Files:**
  - `src/backend/models/phi.rs` (safetensors)
  - `src/backend/models/quantized_phi.rs` (GGUF)
- **Status:** ✅ Complete

#### 4. Qwen Family
- **Variants:** Qwen2, Qwen2.5
- **Formats:** Safetensors ✅, GGUF ✅
- **Downloads:** 94.2M+ (Qwen2.5-1.5B-Instruct)
- **Files:**
  - `src/backend/models/qwen.rs` (safetensors)
  - `src/backend/models/quantized_qwen.rs` (GGUF)
- **Status:** ✅ Complete

#### 5. Gemma Family (Partial)
- **Variants:** Gemma, Gemma 2, Gemma 3
- **Formats:** Safetensors ❌, GGUF ✅
- **Downloads:** High (Google)
- **Files:**
  - `src/backend/models/quantized_gemma.rs` (GGUF only)
- **Status:** 🟡 GGUF only - **needs safetensors support**

---

### 🔥 Priority 1: MVP Critical (Implement Next)

#### 6. DeepSeek Family
- **Variants:** DeepSeek-R1, DeepSeek-V2
- **Formats:** Safetensors ❌, GGUF ❌
- **Downloads:** 421K+ (DeepSeek-R1) - **Trending #1 on HuggingFace**
- **Candle Support:** ✅ YES (`candle-transformers/src/models/deepseek2.rs`)
- **Effort:** MEDIUM (2-3 days)
- **Files to create:**
  - `src/backend/models/deepseek.rs` (safetensors)
  - `src/backend/models/quantized_deepseek.rs` (GGUF)
- **Status:** ❌ Not implemented - **HIGHEST PRIORITY**

#### 7. Gemma (Safetensors)
- **Action:** Complete existing GGUF support
- **Formats:** Safetensors ❌, GGUF ✅
- **Candle Support:** ✅ YES (`candle-transformers/src/models/gemma.rs`)
- **Effort:** LOW (1-2 days)
- **Files to create:**
  - `src/backend/models/gemma.rs` (safetensors)
- **Status:** 🟡 GGUF only - **needs safetensors**

#### 8. Mixtral (MoE)
- **Variants:** Mixtral-8x7B
- **Formats:** Safetensors ❌, GGUF ❌
- **Downloads:** High (Mistral AI)
- **Candle Support:** ✅ YES (`candle-transformers/src/models/mixtral.rs`)
- **Effort:** MEDIUM (2-3 days)
- **Files to create:**
  - `src/backend/models/mixtral.rs` (safetensors)
  - `src/backend/models/quantized_mixtral.rs` (GGUF)
- **Status:** ❌ Not implemented - **MoE differentiator**

---

### 🎯 Priority 2: Post-MVP

#### 9. Yi Family
- **Variants:** Yi-6B, Yi-34B
- **Formats:** Safetensors ❌, GGUF ❌
- **Downloads:** 7.96K+
- **Candle Support:** ✅ YES (`candle-transformers/src/models/yi.rs`)
- **Effort:** MEDIUM (2-3 days)
- **Status:** ❌ Not implemented

#### 10. Starcoder2 Family
- **Variants:** Starcoder2-3B, Starcoder2-7B, Starcoder2-15B
- **Formats:** Safetensors ❌, GGUF ❌
- **Downloads:** Moderate (code generation specialist)
- **Candle Support:** ✅ YES (`candle-transformers/src/models/starcoder2.rs`)
- **Effort:** MEDIUM (2-3 days)
- **Status:** ❌ Not implemented

#### 11. Falcon Family
- **Variants:** Falcon-7B, Falcon-40B
- **Formats:** Safetensors ❌, GGUF ❌
- **Downloads:** Moderate
- **Candle Support:** ✅ YES (`candle-transformers/src/models/falcon.rs`)
- **Effort:** MEDIUM (2-3 days)
- **Status:** ❌ Not implemented

#### 12. Stable-LM Family
- **Variants:** Stable-LM-3B, Stable-LM-7B
- **Formats:** Safetensors ❌, GGUF ❌
- **Downloads:** Moderate (Stability AI)
- **Candle Support:** ✅ YES (`candle-transformers/src/models/stable_lm.rs`)
- **Effort:** MEDIUM (2-3 days)
- **Status:** ❌ Not implemented

---

### 🔮 Priority 3: Future/Experimental

#### 13. Mamba Family
- **Variants:** Mamba-130M, Mamba-370M, Mamba-790M, Mamba-1.4B, Mamba-2.8B
- **Formats:** Safetensors ❌, GGUF ❌
- **Downloads:** Low (experimental)
- **Architecture:** State-space models (alternative to transformers)
- **Candle Support:** ✅ YES (`candle-transformers/src/models/mamba.rs`)
- **Effort:** HIGH (different architecture)
- **Status:** ❌ Not implemented

#### 14. RWKV Family
- **Variants:** RWKV-v5, RWKV-v6
- **Formats:** Safetensors ❌, GGUF ❌
- **Downloads:** Low (niche)
- **Architecture:** RNN-based alternative to transformers
- **Candle Support:** ✅ YES (`candle-transformers/src/models/rwkv_v5.rs`, `rwkv_v6.rs`)
- **Effort:** HIGH (different architecture)
- **Status:** ❌ Not implemented

#### 15. Olmo Family
- **Variants:** Olmo-1B, Olmo-7B, Olmo2
- **Formats:** Safetensors ❌, GGUF ❌
- **Downloads:** Moderate (Allen Institute)
- **Candle Support:** ✅ YES (`candle-transformers/src/models/olmo.rs`, `olmo2.rs`)
- **Effort:** MEDIUM (2-3 days)
- **Status:** ❌ Not implemented

---

### 🔍 Needs Research (Unknown Architecture)

#### 16. Kimi Family (Moonshot AI)
- **Variants:** Kimi-K2-Thinking, Kimi-Linear-48B, Kimi-K2-Instruct
- **Formats:** Unknown
- **Downloads:** 89.5K+ (Kimi-K2-Thinking), 277K+ (Kimi-Linear-48B)
- **Architecture:** Unknown (possibly Llama-based?)
- **Candle Support:** 🔍 UNKNOWN - needs research
- **Effort:** HIGH (architecture unknown)
- **Status:** 🔍 Needs research

#### 17. GPT-OSS (OpenAI)
- **Variants:** GPT-OSS-20B
- **Formats:** Unknown
- **Downloads:** 4.76M+
- **Architecture:** Unknown
- **Candle Support:** 🔍 UNKNOWN - needs research
- **Effort:** HIGH (architecture unknown)
- **Status:** 🔍 Needs research

#### 18. MiniMaxAI/MiniMax-M2
- **Variants:** MiniMax-M2
- **Formats:** Unknown
- **Downloads:** 886K+
- **Architecture:** Unknown
- **Candle Support:** 🔍 UNKNOWN - needs research
- **Effort:** HIGH (architecture unknown)
- **Status:** 🔍 Needs research

---

### 📝 Already Compatible (Documentation Only)

#### 19. SmolLM Family
- **Variants:** SmolLM-135M, SmolLM-360M, SmolLM-1.7B, SmolLM2, SmolLM3-3B
- **Formats:** Uses Llama architecture
- **Downloads:** 57.6K+ (SmolLM3-3B)
- **Architecture:** Llama-based
- **Candle Support:** ✅ YES (uses Llama loader)
- **Effort:** NONE (just documentation)
- **Status:** 📝 Already compatible via Llama - **just document**

---

## Summary Statistics

### Current Support
- **Total Families:** 5 (Llama, Mistral, Phi, Qwen, Gemma)
- **Total Architectures:** 8 (including quantized variants)
- **Safetensors Support:** 4/5 families (80%)
- **GGUF Support:** 5/5 families (100%)
- **Coverage:** ~60% of popular HuggingFace models

### After MVP (Priority 1)
- **Total Families:** 7 (+ DeepSeek, Mixtral)
- **Total Architectures:** 12 (including quantized variants)
- **Safetensors Support:** 7/7 families (100%)
- **GGUF Support:** 7/7 families (100%)
- **Coverage:** ~90% of popular HuggingFace models

### After Priority 2
- **Total Families:** 11 (+ Yi, Starcoder2, Falcon, Stable-LM)
- **Total Architectures:** 18 (including quantized variants)
- **Coverage:** ~95% of popular HuggingFace models

---

## Implementation Effort Summary

| Priority | Models | Total Effort | Expected Impact |
|----------|--------|--------------|-----------------|
| **P1 (MVP)** | DeepSeek, Gemma (safetensors), Mixtral | 1-2 weeks | 90% coverage |
| **P2 (Post-MVP)** | Yi, Starcoder2, Falcon, Stable-LM | 2-3 weeks | 95% coverage |
| **P3 (Future)** | Mamba, RWKV, Olmo | 3-4 weeks | 98% coverage |
| **Research** | Kimi, GPT-OSS, MiniMax-M2 | Unknown | Unknown |

---

## Next Actions

1. ✅ **Approve this plan**
2. 🔥 **TEAM-482:** Implement DeepSeek-R1 (Priority 1, highest impact)
3. 🔥 **TEAM-483:** Add Gemma safetensors support (Priority 1, low effort)
4. 🔥 **TEAM-484:** Implement Mixtral MoE (Priority 1, differentiator)
5. 📝 **TEAM-485:** Document SmolLM compatibility (already works)

---

**Status:** ✅ PLANNING COMPLETE  
**Last Updated:** 2025-11-12  
**Next Review:** After MVP implementation
