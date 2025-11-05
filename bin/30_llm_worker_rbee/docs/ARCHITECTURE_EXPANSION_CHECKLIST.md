# Architecture Expansion Checklist - llm-worker-rbee

**Created by:** TEAM-406  
**Date:** 2025-11-05  
**Purpose:** Roadmap to match Ollama/LM Studio architecture support  
**Status:** 🎯 PLANNING

---

## 🎯 Mission

Expand llm-worker-rbee from **4 architectures** (Llama, Mistral, Phi, Qwen) to **competitive parity** with Ollama and LM Studio.

**Current State:**
- ✅ Llama (tested on all backends)
- ⚠️ Mistral, Phi, Qwen (code ready, needs SafeTensors models)
- ❌ 50+ other architectures available in Candle

**Target State:**
- ✅ 10-15 core architectures (match Ollama/LM Studio baseline)
- ✅ All tested on CPU, CUDA, Metal
- ✅ GGUF support (critical for market)
- ✅ SafeTensors support (current)

---

## 📊 Competitive Analysis

### Ollama Supports (100+ models)
- ✅ Llama (all versions)
- ✅ Mistral (including Mixtral MoE)
- ✅ Gemma (1, 2, 3)
- ✅ Phi (3, 4)
- ✅ Qwen (2, 2.5, 3)
- ✅ DeepSeek (R1, V2, V3, Coder)
- ✅ CodeLlama, StarCoder
- ✅ Granite
- ✅ Falcon
- ✅ Yi

### LM Studio Supports (50+ models)
- Same as Ollama +
- ✅ GPT-OSS (OpenAI)
- ✅ Magistral (reasoning)
- ✅ Gemma-3n (on-device)

### Candle-Transformers Has (113 modules!)
**Text Generation (LLMs):**
- llama, llama2_c
- mistral, mixtral
- phi, phi3
- qwen2, qwen2_moe, qwen3, qwen3_moe
- gemma, gemma2, gemma3
- deepseek2
- falcon
- mpt
- yi
- olmo, olmo2
- starcoder2
- bigcode
- mamba
- granite, granitemoehybrid
- glm4, glm4_new
- chatglm
- codegeex4_9b
- stable_lm
- persimmon
- modernbert
- helium
- based
- rwkv_v5, rwkv_v6
- recurrent_gemma

**Quantized Variants:**
- quantized_llama
- quantized_mistral
- quantized_phi, quantized_phi3
- quantized_qwen2, quantized_qwen3
- quantized_gemma3
- quantized_mpt
- quantized_stable_lm
- quantized_rwkv_v5, quantized_rwkv_v6
- quantized_recurrent_gemma

**Vision Models (for future):**
- llava, moondream, paligemma, pixtral, qwen3_vl, colpali

---

## 🚀 Phase 1: Critical Gap - GGUF Support (HIGHEST PRIORITY)

**Why:** Both Ollama and LM Studio use GGUF as PRIMARY format. SafeTensors-only is a critical competitive gap.

### Tasks

- [ ] **1.1: Add GGUF Loader for Mistral**
  - **Source:** `reference/candle/candle-transformers/src/models/quantized_mistral.rs`
  - **Target:** `bin/30_llm_worker_rbee/src/backend/models/quantized_mistral.rs`
  - **Effort:** 2-3 hours (copy + adapt from Candle)
  - **Benefit:** Unlock 100+ Mistral GGUF models

- [ ] **1.2: Test GGUF Mistral on All Backends**
  - Download: Mistral-7B-Instruct Q4_K_M GGUF
  - Test: CPU, CUDA, Metal
  - Verify: Inference quality, performance

- [ ] **1.3: Update Model Factory for GGUF Mistral**
  - Add `QuantizedMistral` variant to `Model` enum
  - Update `load_model()` to detect Mistral GGUF
  - Update `detect_architecture_from_gguf()` for Mistral

- [ ] **1.4: Document GGUF Support**
  - Update `MODEL_SUPPORT.md` with GGUF status
  - Add GGUF vs SafeTensors comparison
  - Document quantization levels supported

**Acceptance Criteria:**
- ✅ All 4 architectures support BOTH SafeTensors AND GGUF
- ✅ GGUF models tested on all backends
- ✅ Documentation updated

---

## 🎯 Phase 2: Core Architecture Expansion (Match Ollama Baseline)

**Goal:** Add 6 more architectures to reach 10 total (competitive baseline)

### Priority 1: Gemma (Google - Very Popular)

- [ ] **2.1: Add Gemma SafeTensors Support**
  - **Source:** `reference/candle/candle-transformers/src/models/gemma.rs`
  - **Target:** `bin/30_llm_worker_rbee/src/backend/models/gemma.rs`
  - **Effort:** 3-4 hours
  - **Models:** Gemma-2B, Gemma-7B

- [ ] **2.2: Add Gemma2 SafeTensors Support**
  - **Source:** `reference/candle/candle-transformers/src/models/gemma2.rs`
  - **Target:** `bin/30_llm_worker_rbee/src/backend/models/gemma2.rs`
  - **Effort:** 3-4 hours
  - **Models:** Gemma2-9B, Gemma2-27B

- [ ] **2.3: Add Gemma3 GGUF Support**
  - **Source:** `reference/candle/candle-transformers/src/models/quantized_gemma3.rs`
  - **Target:** `bin/30_llm_worker_rbee/src/backend/models/quantized_gemma3.rs`
  - **Effort:** 2-3 hours
  - **Models:** Gemma3-1B, Gemma3-4B, Gemma3-12B, Gemma3-27B

- [ ] **2.4: Test Gemma on All Backends**
  - CPU, CUDA, Metal
  - SafeTensors + GGUF

### Priority 2: DeepSeek (Trending - Reasoning Models)

- [ ] **2.5: Add DeepSeek2 SafeTensors Support**
  - **Source:** `reference/candle/candle-transformers/src/models/deepseek2.rs`
  - **Target:** `bin/30_llm_worker_rbee/src/backend/models/deepseek2.rs`
  - **Effort:** 4-5 hours (MoE architecture, more complex)
  - **Models:** DeepSeek-V2, DeepSeek-Coder-V2

- [ ] **2.6: Test DeepSeek on All Backends**
  - CPU, CUDA, Metal
  - Note: Large models, may need quantization

### Priority 3: Falcon (Popular Alternative)

- [ ] **2.7: Add Falcon SafeTensors Support**
  - **Source:** `reference/candle/candle-transformers/src/models/falcon.rs`
  - **Target:** `bin/30_llm_worker_rbee/src/backend/models/falcon.rs`
  - **Effort:** 3-4 hours
  - **Models:** Falcon-7B, Falcon-40B

- [ ] **2.8: Test Falcon on All Backends**

### Priority 4: StarCoder2 (Coding Models)

- [ ] **2.9: Add StarCoder2 SafeTensors Support**
  - **Source:** `reference/candle/candle-transformers/src/models/starcoder2.rs`
  - **Target:** `bin/30_llm_worker_rbee/src/backend/models/starcoder2.rs`
  - **Effort:** 3-4 hours
  - **Models:** StarCoder2-3B, StarCoder2-7B, StarCoder2-15B

- [ ] **2.10: Test StarCoder2 on All Backends**

### Priority 5: Yi (Chinese + English)

- [ ] **2.11: Add Yi SafeTensors Support**
  - **Source:** `reference/candle/candle-transformers/src/models/yi.rs`
  - **Target:** `bin/30_llm_worker_rbee/src/backend/models/yi.rs`
  - **Effort:** 3-4 hours
  - **Models:** Yi-6B, Yi-34B

- [ ] **2.12: Test Yi on All Backends**

### Priority 6: Granite (IBM - Enterprise)

- [ ] **2.13: Add Granite SafeTensors Support**
  - **Source:** `reference/candle/candle-transformers/src/models/granite.rs`
  - **Target:** `bin/30_llm_worker_rbee/src/backend/models/granite.rs`
  - **Effort:** 3-4 hours
  - **Models:** Granite-3B, Granite-7B, Granite-20B

- [ ] **2.14: Add Granite MoE Hybrid Support**
  - **Source:** `reference/candle/candle-transformers/src/models/granitemoehybrid.rs`
  - **Target:** `bin/30_llm_worker_rbee/src/backend/models/granitemoehybrid.rs`
  - **Effort:** 4-5 hours (MoE architecture)

- [ ] **2.15: Test Granite on All Backends**

**Phase 2 Summary:**
- **Total Architectures:** 10 (4 existing + 6 new)
- **Total Effort:** ~40-50 hours
- **Competitive Status:** ✅ Match Ollama/LM Studio baseline

---

## 🚀 Phase 3: Advanced Features (Match Ollama/LM Studio Parity)

### Mixtral (MoE - Mixture of Experts)

- [ ] **3.1: Add Mixtral SafeTensors Support**
  - **Source:** `reference/candle/candle-transformers/src/models/mixtral.rs`
  - **Target:** `bin/30_llm_worker_rbee/src/backend/models/mixtral.rs`
  - **Effort:** 5-6 hours (MoE architecture, complex)
  - **Models:** Mixtral-8x7B, Mixtral-8x22B

- [ ] **3.2: Test Mixtral on All Backends**
  - Note: Very large models, needs quantization

### Qwen MoE (Mixture of Experts)

- [ ] **3.3: Add Qwen2 MoE SafeTensors Support**
  - **Source:** `reference/candle/candle-transformers/src/models/qwen2_moe.rs`
  - **Target:** `bin/30_llm_worker_rbee/src/backend/models/qwen2_moe.rs`
  - **Effort:** 5-6 hours

- [ ] **3.4: Add Qwen3 MoE SafeTensors Support**
  - **Source:** `reference/candle/candle-transformers/src/models/qwen3_moe.rs`
  - **Target:** `bin/30_llm_worker_rbee/src/backend/models/qwen3_moe.rs`
  - **Effort:** 5-6 hours

- [ ] **3.5: Test Qwen MoE on All Backends**

### Phi3 (Microsoft - Latest)

- [ ] **3.6: Add Phi3 SafeTensors Support**
  - **Source:** `reference/candle/candle-transformers/src/models/phi3.rs`
  - **Target:** `bin/30_llm_worker_rbee/src/backend/models/phi3.rs`
  - **Effort:** 3-4 hours
  - **Models:** Phi-3-Mini, Phi-3-Medium

- [ ] **3.7: Add Phi3 GGUF Support**
  - **Source:** `reference/candle/candle-transformers/src/models/quantized_phi3.rs`
  - **Target:** `bin/30_llm_worker_rbee/src/backend/models/quantized_phi3.rs`
  - **Effort:** 2-3 hours

- [ ] **3.8: Test Phi3 on All Backends**

### Olmo/Olmo2 (Open LLM)

- [ ] **3.9: Add Olmo SafeTensors Support**
  - **Source:** `reference/candle/candle-transformers/src/models/olmo.rs`
  - **Target:** `bin/30_llm_worker_rbee/src/backend/models/olmo.rs`
  - **Effort:** 3-4 hours

- [ ] **3.10: Add Olmo2 SafeTensors Support**
  - **Source:** `reference/candle/candle-transformers/src/models/olmo2.rs`
  - **Target:** `bin/30_llm_worker_rbee/src/backend/models/olmo2.rs`
  - **Effort:** 3-4 hours

- [ ] **3.11: Test Olmo on All Backends**

**Phase 3 Summary:**
- **Total Architectures:** 15+ (10 from Phase 2 + 5+ advanced)
- **Total Effort:** ~40-50 hours
- **Competitive Status:** ✅ EXCEED Ollama/LM Studio baseline

---

## 🎯 Phase 4: Specialized Models (Future)

### Coding Models

- [ ] **4.1: BigCode**
  - **Source:** `reference/candle/candle-transformers/src/models/bigcode.rs`
  - **Models:** StarCoder, SantaCoder

- [ ] **4.2: CodeGeeX4**
  - **Source:** `reference/candle/candle-transformers/src/models/codegeex4_9b.rs`
  - **Models:** CodeGeeX4-9B

- [ ] **4.3: MPT**
  - **Source:** `reference/candle/candle-transformers/src/models/mpt.rs`
  - **Models:** MPT-7B, MPT-30B

### Alternative Architectures

- [ ] **4.4: Mamba (State Space Models)**
  - **Source:** `reference/candle/candle-transformers/src/models/mamba.rs`
  - **Models:** Mamba-130M, Mamba-1.4B, Mamba-2.8B

- [ ] **4.5: RWKV v5/v6 (RNN-based)**
  - **Source:** `reference/candle/candle-transformers/src/models/rwkv_v5.rs`
  - **Source:** `reference/candle/candle-transformers/src/models/rwkv_v6.rs`
  - **Models:** RWKV-4-World, RWKV-5-World

- [ ] **4.6: Recurrent Gemma**
  - **Source:** `reference/candle/candle-transformers/src/models/recurrent_gemma.rs`
  - **Models:** RecurrentGemma-2B

### Chinese/Multilingual Models

- [ ] **4.7: ChatGLM**
  - **Source:** `reference/candle/candle-transformers/src/models/chatglm.rs`
  - **Models:** ChatGLM-6B, ChatGLM2-6B

- [ ] **4.8: GLM4**
  - **Source:** `reference/candle/candle-transformers/src/models/glm4.rs`
  - **Models:** GLM-4-9B

---

## 📋 Implementation Pattern (Copy from Candle)

### For Each New Architecture:

1. **Copy Candle Implementation**
   ```bash
   cp reference/candle/candle-transformers/src/models/ARCH.rs \
      bin/30_llm_worker_rbee/src/backend/models/ARCH.rs
   ```

2. **Adapt to rbee Pattern**
   - Add `pub struct ARCHModel { ... }`
   - Implement `load()` method
   - Implement `forward()` method
   - Implement `eos_token_id()`, `vocab_size()`, `reset_cache()`

3. **Update Model Factory**
   - Add variant to `Model` enum in `mod.rs`
   - Add match arm in `forward()`, `eos_token_id()`, etc.
   - Update `detect_architecture()` for new arch
   - Update `load_model()` to handle new arch

4. **Add Quantized Variant (if available)**
   ```bash
   cp reference/candle/candle-transformers/src/models/quantized_ARCH.rs \
      bin/30_llm_worker_rbee/src/backend/models/quantized_ARCH.rs
   ```

5. **Test on All Backends**
   - Download test model (small size)
   - Test CPU: `cargo test --features cpu`
   - Test Metal: `./scripts/homelab/llorch-remote mac.home.arpa metal debug-inference`
   - Test CUDA: `./scripts/homelab/llorch-remote workstation.home.arpa cuda debug-inference`

6. **Update Documentation**
   - Add to `MODEL_SUPPORT.md`
   - Update compatibility matrix
   - Document any architecture-specific quirks

---

## 🎯 Success Metrics

### Phase 1 Complete (GGUF Support)
- ✅ 4 architectures support GGUF
- ✅ All tested on CPU, CUDA, Metal
- ✅ Documentation updated

### Phase 2 Complete (Core Expansion)
- ✅ 10 architectures total
- ✅ Match Ollama/LM Studio baseline
- ✅ All tested on all backends

### Phase 3 Complete (Advanced Features)
- ✅ 15+ architectures total
- ✅ MoE models supported
- ✅ EXCEED Ollama/LM Studio baseline

### Phase 4 Complete (Specialized)
- ✅ 20+ architectures total
- ✅ Coding models supported
- ✅ Alternative architectures (Mamba, RWKV)
- ✅ Multilingual models

---

## 📊 Effort Estimates

| Phase | Architectures | Effort | Priority |
|-------|---------------|--------|----------|
| Phase 1: GGUF | 4 (existing) | 10-15 hours | 🔥 CRITICAL |
| Phase 2: Core | +6 (total 10) | 40-50 hours | 🎯 HIGH |
| Phase 3: Advanced | +5 (total 15) | 40-50 hours | ⚠️ MEDIUM |
| Phase 4: Specialized | +5+ (total 20+) | 50-60 hours | 💡 LOW |

**Total to match Ollama/LM Studio:** ~100-115 hours (Phase 1 + Phase 2)

---

## 🚨 Critical Dependencies

### Before Starting:
1. ✅ Candle fork with mask fix (already done)
2. ✅ Model factory pattern (already done)
3. ✅ Backend abstraction (CPU/CUDA/Metal) (already done)

### For GGUF Support (Phase 1):
1. ❌ GGUF loader for Mistral (copy from Candle)
2. ❌ Test GGUF models on all backends

### For Each New Architecture:
1. ❌ SafeTensors test model (download from HuggingFace)
2. ❌ GGUF test model (download from HuggingFace)
3. ❌ Tokenizer support (usually included in Candle)

---

## 📝 Notes

### Why Candle-Transformers is Perfect for This:
1. **Already Implemented:** 113 model modules ready to copy
2. **Well-Tested:** Used in production by HuggingFace
3. **Consistent API:** All models follow same pattern
4. **Quantization Support:** GGUF loaders already exist
5. **Backend Agnostic:** Works on CPU, CUDA, Metal

### Candle vs rbee Differences:
- **Candle:** Library with examples
- **rbee:** Production worker with HTTP API
- **Adaptation Needed:** Wrap Candle models in rbee's `Model` enum

### Key Files to Track:
- `bin/30_llm_worker_rbee/src/backend/models/mod.rs` - Model factory
- `bin/30_llm_worker_rbee/docs/MODEL_SUPPORT.md` - Documentation
- `reference/candle/candle-transformers/src/models/` - Source implementations

---

**TEAM-406 - Architecture Expansion Checklist**  
**Next:** Start with Phase 1 (GGUF Support) - CRITICAL for market competitiveness
