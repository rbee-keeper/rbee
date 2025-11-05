# TEAM-406: Competitive Research - Ollama & LM Studio Compatibility

**Created:** 2025-11-05  
**Team:** TEAM-406  
**Status:** 🔬 RESEARCH PHASE  
**Purpose:** Define compatibility standards to match/exceed Ollama and LM Studio

---

## 🎯 Research Objectives

1. **Ollama Compatibility Standards**
   - Model architectures supported
   - File formats supported
   - Quantization levels supported
   - Generation parameters exposed
   - User-facing compatibility communication

2. **LM Studio Compatibility Standards**
   - Model architectures supported
   - File formats supported
   - Quantization levels supported
   - Generation parameters exposed
   - User-facing compatibility communication

3. **Industry Baseline**
   - Minimum viable compatibility matrix
   - Table stakes generation parameters
   - Standard model formats
   - Common quantization levels

---

## 📊 Ollama Compatibility Analysis

### Research Tasks
- [ ] Visit Ollama GitHub repo (https://github.com/ollama/ollama)
- [ ] Read Ollama model library documentation
- [ ] Check Ollama supported models list
- [ ] Analyze Ollama API parameters
- [ ] Review Ollama compatibility matrix (if exists)
- [ ] Check Ollama format support (GGUF, SafeTensors, etc.)

### Key Questions
1. **Model Architectures:**
   - Which architectures does Ollama support? (Llama, Mistral, Phi, Qwen, etc.)
   - Are there architecture-specific limitations?
   - How does Ollama communicate architecture support to users?

2. **File Formats:**
   - Primary format: GGUF? SafeTensors? Both?
   - Quantization formats supported (Q4, Q5, Q8, FP16, etc.)
   - Conversion tools provided?

3. **Generation Parameters:**
   - Which parameters are exposed? (temperature, top_p, top_k, etc.)
   - Default values for each parameter?
   - Parameter validation/ranges?

4. **Backend Support:**
   - CPU support?
   - CUDA support?
   - Metal support?
   - ROCm support?
   - Multi-GPU support?

5. **User Communication:**
   - How does Ollama show compatibility?
   - Model cards? Compatibility matrix? Tags?
   - Error messages when incompatible?

### Findings (TEAM-406 Research Complete)

## Supported Architectures
- ✅ Llama (including Llama 2, Llama 3, Llama 3.1, and Llama 3.2)
- ✅ Mistral (including Mistral 1, Mistral 2, and Mixtral)
- ✅ Gemma (including Gemma 1 and Gemma 2)
- ✅ Phi3
- ✅ Qwen (multiple versions in library)
- ✅ DeepSeek (R1, V2, V3, Coder)
- ✅ CodeLlama, StarCoder, Granite, Falcon, Yi, and 100+ others

**Source:** https://ollama.com/library and https://github.com/ollama/ollama/blob/main/docs/modelfile.md

## File Formats
- **Primary:** GGUF (native format)
- **Secondary:** Safetensors (supported via import)
- **Quantization Levels Supported:**
  - Q4_0, Q4_1 (4-bit quantization)
  - Q5_0, Q5_1 (5-bit quantization)
  - Q8_0 (8-bit quantization)
  - K-means quantizations: Q3_K_S, Q3_K_M, Q3_K_L, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K
  - FP16 (full precision)

**Key Insight:** Ollama is GGUF-first. Users can import Safetensors, but GGUF is the standard.

## Generation Parameters (Complete List)
**Standard Parameters:**
- `temperature` (default: 0.8) - Creativity/randomness (0.0-2.0)
- `top_p` (default: 0.9) - Nucleus sampling (0.0-1.0)
- `top_k` (default: 40) - Top-k sampling
- `repeat_penalty` (default: 1.1) - Penalize repeated tokens
- `repeat_last_n` (default: 64) - Window for repeat penalty
- `num_predict` (default: 128) - Max tokens to generate
- `num_ctx` (default: 2048) - Context window size
- `seed` - RNG seed for reproducibility
- `stop` - Stop sequences (array of strings)

**Advanced Parameters:**
- `min_p` - Minimum probability threshold
- `tfs_z` - Tail-free sampling
- `mirostat` - Mirostat sampling mode (0, 1, 2)
- `mirostat_eta` - Mirostat learning rate
- `mirostat_tau` - Mirostat target entropy

**Total:** 14+ generation parameters exposed

**Source:** https://technovangelist.com/notes/ollama-parameters and https://github.com/ollama/ollama/blob/main/docs/api.md

## Backend Support
- ✅ **CPU:** Full support (all architectures)
- ✅ **CUDA:** Full support (NVIDIA GPUs)
- ✅ **Metal:** Full support (Apple Silicon M1/M2/M3)
- ✅ **ROCm:** Supported (AMD GPUs)
- ✅ **Multi-GPU:** Supported (automatic distribution)

**RAM Requirements:**
- 7B models: 8 GB RAM minimum
- 13B models: 16 GB RAM minimum
- 33B models: 32 GB RAM minimum

## Compatibility Communication
**Method:** Implicit via model library
- **Model Library:** https://ollama.com/library shows all available models
- **Model Cards:** Each model has size variants (1b, 7b, 13b, 70b, etc.)
- **Quantization Tags:** Models show quantization level in filename (e.g., `Q4_K_M`)
- **No Explicit Compatibility Matrix:** Users must know their hardware capabilities
- **Error Messages:** Runtime errors if model too large for available RAM

**User Experience:**
- Simple: `ollama run llama3.2` (auto-selects appropriate size)
- Explicit: `ollama run llama3.2:70b` (user specifies size)
- No pre-install compatibility checks
- No warnings about hardware limitations until runtime

---

## 📊 LM Studio Compatibility Analysis

### Research Tasks
- [ ] Visit LM Studio website (https://lmstudio.ai/)
- [ ] Read LM Studio documentation
- [ ] Check LM Studio supported models
- [ ] Analyze LM Studio UI for compatibility indicators
- [ ] Review LM Studio model search/filter features
- [ ] Check LM Studio format support

### Key Questions
1. **Model Architectures:**
   - Which architectures does LM Studio support?
   - Architecture-specific features?
   - How is architecture communicated in UI?

2. **File Formats:**
   - Primary format support?
   - Quantization support?
   - Download/conversion tools?

3. **Generation Parameters:**
   - Which parameters in UI?
   - Advanced vs simple mode?
   - Parameter presets?

4. **Backend Support:**
   - CPU support?
   - GPU support (CUDA, Metal, etc.)?
   - Multi-GPU?

5. **User Communication:**
   - Compatibility indicators in model browser?
   - Warnings for incompatible models?
   - Performance estimates?

### Findings (TEAM-406 Research Complete)

## Supported Architectures
- ✅ Llama (all versions including Llama 4)
- ✅ Mistral (including Mistral-Nemo, Mistral-Small, Magistral)
- ✅ Phi (Phi-3, Phi-4, Phi-4-reasoning, Phi-4-mini)
- ✅ Qwen (Qwen3, Qwen3-VL, Qwen3-Coder - multiple sizes)
- ✅ Gemma (Gemma-3, Gemma-3n)
- ✅ DeepSeek (DeepSeek-R1, DeepSeek-V3, DeepSeek-Coder)
- ✅ Granite, Codestral, GPT-OSS, and 50+ other architectures

**Source:** https://lmstudio.ai/models (comprehensive model catalog)

## File Formats
- **Primary:** GGUF (native format)
- **Secondary:** MLX (for Apple Silicon), Safetensors (via import)
- **Quantization Levels Supported:**
  - Same as Ollama: Q3_K, Q4_K, Q5_K, Q6_K, Q8_0
  - MLX-specific: 4-bit, 8-bit quantizations
  - FP16 (full precision)

**Key Insight:** LM Studio supports BOTH GGUF and MLX formats, with automatic format selection based on hardware (MLX for Apple Silicon, GGUF for others).

## Generation Parameters
**Standard Parameters (UI Exposed):**
- `temperature` - Creativity/randomness
- `maxTokens` / `num_predict` - Maximum output tokens
- `topP` - Nucleus sampling
- `topK` - Top-k sampling
- `repeatPenalty` - Penalize repeated tokens
- `contextLength` - Context window size
- `seed` - RNG seed for reproducibility

**Advanced Parameters (Power User / Developer Mode):**
- GPU offload ratio
- Batch size
- Thread count
- Structured outputs (JSON schema enforcement)
- Stop sequences
- System prompts

**Presets System:**
- Save parameter combinations as named presets
- Import/export presets
- Share presets via LM Studio Hub
- Per-chat preset association

**Total:** 10+ generation parameters with preset management

**Source:** https://lmstudio.ai/docs/app/presets and https://lmstudio.ai/docs/typescript/llm-prediction/parameters

## Backend Support
- ✅ **CPU:** Full support (x64, ARM, ARM64)
- ✅ **CUDA:** Full support (NVIDIA GPUs)
- ✅ **Metal:** Full support (Apple Silicon M1/M2/M3/M4)
- ✅ **MLX:** Optimized for Apple Silicon (macOS 14.0+)
- ✅ **Multi-GPU:** Supported (GPU offload ratio configurable)

**RAM Requirements:**
- 8GB RAM: Small models only (with warnings)
- 16GB RAM: Recommended minimum
- 4GB+ VRAM: Recommended for GPU acceleration

**Platform Support:**
- macOS: Apple Silicon only (Intel not supported)
- Windows: x64 and ARM (Snapdragon X Elite)
- Linux: x64 and ARM64 (Ubuntu 20.04+)

## Compatibility Communication
**Method:** Rich UI with compatibility indicators

**Model Browser Features:**
- **RAM/VRAM Estimates:** Shows estimated memory usage per model variant
- **Quantization Badges:** Clear labels (Q4_K_M, Q8_0, MLX-4bit, etc.)
- **Size Variants:** Multiple sizes per model (1B, 3B, 7B, 14B, 30B, etc.)
- **Performance Indicators:** Downloads count, update recency
- **Compatibility Types:** Shows supported formats (GGUF, Safetensors, MLX)
- **Architecture Tags:** Clear architecture labels (Llama, Mistral, etc.)

**Search & Filtering:**
- **Basic Keyword Search:** Search by model name, architecture, or HuggingFace user/repo
- **Format Filter:** GGUF vs MLX checkbox (hardware-based)
- **NO Advanced Filters:** Architecture, parameters, quants filtering is a FEATURE REQUEST (not implemented)
- **Source:** Direct HuggingFace API search (no custom filtering backend)

**User Experience:**
- **Pre-Download Checks:** Shows RAM/VRAM requirements before download
- **Smart Defaults:** Auto-selects appropriate quantization for hardware
- **My Models View:** Shows loaded models with memory usage
- **Error Prevention:** Warns if model too large for available RAM/VRAM
- **Model.yaml Metadata:** Rich metadata for each model (architectures, compatibilityTypes, minMemoryUsageBytes)

**Key Advantage:** LM Studio provides EXPLICIT compatibility information BEFORE download, unlike Ollama's runtime-only errors.

**Key Limitation:** Search is basic keyword-only. Advanced filtering (by architecture, parameters, quants) is REQUESTED but not implemented.

**Source:** https://lmstudio.ai/docs/app/modelyaml, https://lmstudio.ai/docs/app/system-requirements, https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/617

---

## 📊 Industry Baseline (OpenAI, Anthropic, llama.cpp)

### OpenAI API Parameters (Reference)
**Source:** `bin/.specs/.docs/GENERATION_PARAMETERS_ANALYSIS.md`

**Standard Parameters:**
- `temperature` (0.0-2.0) - Sampling temperature
- `top_p` (0.0-1.0) - Nucleus sampling
- `frequency_penalty` (-2.0 to 2.0) - Penalize frequent tokens
- `presence_penalty` (-2.0 to 2.0) - Penalize present tokens
- `max_tokens` (1-∞) - Maximum output tokens
- `stop` (string[]) - Stop sequences
- `logit_bias` (map) - Bias specific tokens
- `seed` (int) - Deterministic sampling
- `n` (int) - Number of completions
- `stream` (bool) - SSE streaming

**Total:** 10 parameters

### llama.cpp / Ollama / LM Studio Parameters
**Common Parameters:**
- `temperature` - Sampling temperature
- `top_p` - Nucleus sampling
- `top_k` - Top-k sampling
- `repeat_penalty` / `repetition_penalty` - Penalize repeated tokens
- `min_p` - Minimum probability threshold
- `tfs_z` - Tail-free sampling
- `typical_p` - Locally typical sampling

### rbee Current State
**Implemented (M0-W-1300):**
- ✅ `temperature` (0.0-2.0)
- ✅ `max_tokens` (1-2048)
- ✅ `seed` (uint64)

**Missing:**
- ❌ `top_p` (defined but not exposed)
- ❌ `top_k` (defined but not exposed)
- ❌ `frequency_penalty`
- ❌ `presence_penalty`
- ❌ `repeat_penalty`
- ❌ `min_p`
- ❌ `stop` sequences

**Gap:** 7+ missing standard parameters

---

## 🎯 Ideal rbee Compatibility Matrix Specification

### Minimum Viable Compatibility (MVP)

#### 1. Model Metadata (Required)
```typescript
interface ModelMetadata {
  // Architecture
  architecture: 'llama' | 'mistral' | 'phi' | 'qwen' | 'gemma' | 'unknown'
  
  // Format
  format: 'safetensors' | 'gguf' | 'pytorch'
  
  // Quantization (if applicable)
  quantization?: 'fp16' | 'fp32' | 'q4_0' | 'q4_1' | 'q5_0' | 'q5_1' | 'q8_0'
  
  // Size
  parameters: string // e.g., "7B", "13B", "70B"
  size_bytes: number
  
  // Context
  max_context_length: number
}
```

#### 2. Worker Capabilities (Required)
```typescript
interface WorkerCapabilities {
  // Backend
  worker_type: 'cpu' | 'cuda' | 'metal'
  
  // Supported architectures
  supported_architectures: string[] // ['llama', 'mistral', 'phi', 'qwen']
  
  // Supported formats
  supported_formats: string[] // ['safetensors', 'gguf']
  
  // Context limits
  max_context_length: number
  
  // Features
  supports_streaming: boolean
  supports_batching: boolean
  
  // Generation parameters
  supported_parameters: {
    temperature: { min: number, max: number, default: number }
    top_p?: { min: number, max: number, default: number }
    top_k?: { min: number, max: number, default: number }
    max_tokens: { min: number, max: number, default: number }
    // ... more parameters
  }
}
```

#### 3. Compatibility Check Function
```typescript
interface CompatibilityResult {
  compatible: boolean
  reasons: string[] // Why compatible or not
  warnings: string[] // Potential issues
  recommendations: string[] // Better alternatives
}

function checkCompatibility(
  model: ModelMetadata,
  worker: WorkerCapabilities
): CompatibilityResult
```

#### 4. User-Facing Display
**Model Detail Page:**
- ✅ Compatible workers (green badges)
- ⚠️ Partially compatible workers (yellow badges with warnings)
- ❌ Incompatible workers (red badges with reasons)

**Worker Selection:**
- Show compatible models for selected worker
- Filter models by worker compatibility
- Performance estimates (if available)

---

## 📋 Research Action Items

### Immediate Tasks (TEAM-406)
- [x] Research Ollama compatibility (2-3 hours) ✅ COMPLETE
  - [x] Visit GitHub repo
  - [x] Read model library docs
  - [x] Check supported models list
  - [x] Document findings above
  
- [x] Research LM Studio compatibility (2-3 hours) ✅ COMPLETE
  - [x] Visit website
  - [x] Read documentation
  - [x] Check model browser
  - [x] Document findings above
  
- [x] Define rbee ideal compatibility matrix (1-2 hours) ✅ COMPLETE
  - [x] Based on Ollama findings
  - [x] Based on LM Studio findings
  - [x] Based on industry standards
  - [x] Write specification below

### Deliverables
- [x] Completed Ollama findings section ✅
- [x] Completed LM Studio findings section ✅
- [x] Completed ideal compatibility spec ✅
- [x] Recommendations for rbee implementation ✅

---

## 🎯 Recommendations (TEAM-406 Research Complete)

### Priority 1: Must-Have Features (Match Industry Baseline)

1. **Explicit Compatibility Matrix**
   - Show which workers can run which models BEFORE download/install
   - Learn from LM Studio: Show RAM/VRAM estimates per model
   - Improve on Ollama: Don't wait for runtime errors

2. **Architecture Support**
   - ✅ Llama (already tested)
   - ⚠️ Mistral, Phi, Qwen (code ready, needs testing)
   - Add: Gemma, DeepSeek (high priority based on popularity)
   - Total target: 6-8 core architectures

3. **Format Support**
   - ✅ SafeTensors (current primary)
   - ❌ GGUF (critical gap - both Ollama and LM Studio use GGUF as primary)
   - Priority: Add GGUF support OR clearly communicate SafeTensors-only limitation

4. **Generation Parameters (Minimum Viable)**
   - ✅ temperature (already have)
   - ✅ max_tokens (already have)
   - ✅ seed (already have)
   - ❌ top_p (CRITICAL - both competitors have it)
   - ❌ top_k (CRITICAL - both competitors have it)
   - ❌ repeat_penalty (CRITICAL - both competitors have it)
   - Target: 6-8 parameters minimum

5. **Pre-Install Compatibility Checks**
   - Check architecture compatibility before install
   - Check format compatibility before install
   - Check RAM/context length requirements
   - Show clear error messages with suggestions

---

### Priority 2: Should-Have Features (Competitive Parity)

1. **Quantization Support**
   - Document which quantization levels work with rbee workers
   - If GGUF support added: Q4_K_M, Q5_K_M, Q8_0 (most common)
   - Show quantization level in model metadata

2. **Context Length Handling**
   - Current: 32,768 tokens max (good baseline)
   - Ollama default: 2,048 tokens (we're better!)
   - LM Studio: Configurable (we should match)
   - Show context length in compatibility matrix

3. **Backend-Specific Compatibility (Pure Isolation)**
   - **CPU workers:** CPU-only execution (no GPU offloading)
   - **CUDA workers:** CUDA-only execution (no CPU fallback)
   - **Metal workers:** Metal-only execution (no CPU fallback)
   - **Architecture support:** Same across all backends (if SafeTensors available)
   - **Key principle:** CUDA === CUDA only (no mixed execution)

4. **Model Metadata Extraction**
   - Extract architecture from HuggingFace tags
   - Extract format from file list
   - Extract quantization from filename
   - Extract context length from config.json
   - Cache metadata for performance

5. **Compatibility Confidence Levels**
   - **High:** Tested and verified (e.g., Llama on all backends)
   - **Medium:** Code ready, should work (e.g., Mistral, Phi, Qwen)
   - **Low:** Untested, might work
   - **None:** Incompatible (wrong format, architecture, etc.)

---

### Priority 3: Nice-to-Have Features (Competitive Advantages)

1. **Multi-Machine Compatibility**
   - rbee's UNIQUE advantage: Multi-machine orchestration
   - Show which models can run across distributed workers
   - Recommend optimal worker distribution for large models

2. **Heterogeneous Hardware Support**
   - rbee's UNIQUE advantage: Mix CUDA + Metal + CPU
   - Show compatibility across mixed hardware setups
   - Suggest best worker for each model based on available hardware

3. **Advanced Parameter Presets**
   - Learn from LM Studio: Save parameter combinations as presets
   - Presets for: coding, creative writing, reasoning, etc.
   - Import/export presets (JSON format)

4. **Performance Estimates**
   - Estimate tokens/second based on hardware
   - Show expected inference speed per worker
   - Help users choose fastest compatible worker

5. **Compatibility History**
   - Track which model-worker combinations user has successfully used
   - Recommend based on past success
   - Learn from user's hardware capabilities

---

### Competitive Advantages (Where rbee WINS)

1. **Multi-Machine Orchestration**
   - Ollama: ❌ Single machine only
   - LM Studio: ❌ Single machine only
   - rbee: ✅ Distribute across multiple machines via SSH (Queen → Hive)

2. **Heterogeneous Hardware Clusters**
   - Ollama: ❌ One backend per instance
   - LM Studio: ❌ One backend per instance
   - rbee: ✅ Mix CUDA + Metal + CPU workers in ONE cluster

3. **Pure Backend Isolation (No Offloading)**
   - Ollama: ⚠️ CPU fallback, GPU offloading (complexity)
   - LM Studio: ⚠️ GPU offload ratio (complexity)
   - rbee: ✅ **CUDA === CUDA only, Metal === Metal only, CPU === CPU only**
   - **Advantage:** Simpler compatibility (no partial offloading edge cases)

4. **User-Scriptable Routing (Rhai)**
   - Ollama: ❌ No custom routing
   - LM Studio: ❌ No custom routing
   - rbee: ✅ Rhai scripts for intelligent model routing

5. **Web-First Architecture**
   - Ollama: CLI-first
   - LM Studio: Desktop app only
   - rbee: ✅ Web marketplace + Desktop app + API

6. **Explicit Compatibility Matrix**
   - Ollama: ❌ Implicit (runtime errors only)
   - LM Studio: ✅ Good (RAM/VRAM estimates)
   - rbee: ✅ Can be BETTER (show distributed + per-backend compatibility)

7. **Advanced Filtering Opportunity** 🎯 **BIG WIN**
   - Ollama: ❌ No model search/filtering
   - LM Studio: ❌ Basic keyword search only (advanced filtering is [FEATURE REQUEST #617](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/617))
   - rbee: ✅ **Can implement FIRST** (filter by architecture, format, worker compatibility, backend type)
   - **Opportunity:** Beat LM Studio to market with advanced filtering
   - **Filters to implement:**
     - Architecture (Llama, Mistral, Phi, Qwen, etc.)
     - Format (SafeTensors, GGUF future)
     - Parameter size (<=1B, 1B-4B, 4B-9B, 9B-20B, 20B+)
     - Worker compatibility (show only models compatible with installed workers)
     - Backend type (CPU, CUDA, Metal)
     - Context length (<=8K, 8K-32K, 32K+)

---

### Competitive Gaps (Where rbee is BEHIND)

1. **GGUF Format Support**
   - Ollama: ✅ GGUF primary
   - LM Studio: ✅ GGUF + MLX
   - rbee: ❌ SafeTensors only
   - **Impact:** CRITICAL - Most models distributed as GGUF

2. **Generation Parameters**
   - Ollama: ✅ 14+ parameters
   - LM Studio: ✅ 10+ parameters + presets
   - rbee: ⚠️ 3 parameters (temperature, max_tokens, seed)
   - **Impact:** HIGH - Users expect top_p, top_k, repeat_penalty

3. **Model Library Size**
   - Ollama: ✅ 100+ models in library
   - LM Studio: ✅ 50+ models in catalog
   - rbee: ✅ 1000+ models (HuggingFace) but no curation
   - **Impact:** MEDIUM - Need curated "rbee-verified" list

4. **Maturity & Stability**
   - Ollama: ✅ Battle-tested, stable
   - LM Studio: ✅ Polished UI, stable
   - rbee: ⚠️ Pre-1.0, breaking changes OK
   - **Impact:** LOW - Expected for v0.x

5. **Documentation**
   - Ollama: ✅ Excellent docs
   - LM Studio: ✅ Comprehensive docs
   - rbee: ⚠️ Good architecture docs, needs user guides
   - **Impact:** MEDIUM - Need compatibility guide

---

### Implementation Strategy

#### Phase 1: Close Critical Gaps (TEAM-407 to TEAM-409)
1. Add top_p, top_k, repeat_penalty parameters (minimum viable)
2. Create compatibility check function (architecture + format)
3. Extract model metadata from HuggingFace
4. Document SafeTensors-only limitation clearly
5. Add compatibility confidence levels

#### Phase 2: Build Compatibility UI (TEAM-410 to TEAM-411)
1. Show compatibility badges on model pages (green/yellow/red)
2. Add worker filter to model list
3. Show RAM/context length requirements
4. Add pre-install compatibility checks in Keeper
5. Show compatibility warnings with suggestions

#### Phase 3: Leverage Unique Advantages (Future)
1. Show distributed compatibility (multi-machine)
2. Show heterogeneous hardware compatibility
3. Add performance estimates per worker
4. Add Rhai routing examples for compatibility
5. Create "rbee-verified" model list

#### Phase 4: Close Remaining Gaps (Future)
1. Add GGUF support (large effort, high impact)
2. Add more generation parameters (mirostat, min_p, tfs_z)
3. Add parameter presets system
4. Expand architecture support (Gemma, DeepSeek, etc.)
5. Create comprehensive user guides

---

### Recommended Minimum Viable Compatibility Matrix

```typescript
interface ModelMetadata {
  // Identity
  id: string
  name: string
  
  // Compatibility
  architecture: 'llama' | 'mistral' | 'phi' | 'qwen' | 'gemma' | 'unknown'
  format: 'safetensors' | 'gguf' | 'pytorch'
  quantization?: 'fp16' | 'fp32' | 'q4_0' | 'q4_k_m' | 'q5_k_m' | 'q8_0'
  
  // Requirements
  parameters: string // "7B", "13B", "70B"
  size_bytes: number
  max_context_length: number
  min_ram_bytes: number // NEW: Minimum RAM required
}

interface WorkerCapabilities {
  // Identity
  id: string
  worker_type: 'cpu' | 'cuda' | 'metal'
  platform: 'linux' | 'macos' | 'windows'
  
  // Capabilities
  supported_architectures: string[] // ['llama', 'mistral', 'phi', 'qwen']
  supported_formats: string[] // ['safetensors'] (GGUF future)
  max_context_length: number // 32768
  
  // Features
  supports_streaming: boolean
  supports_batching: boolean
  
  // Pure Backend Isolation (rbee-specific)
  pure_backend: boolean // true - CUDA === CUDA only, no CPU fallback
  supports_offloading: boolean // false - no GPU offloading
  
  // Parameters (NEW)
  supported_parameters: {
    temperature: { min: 0.0, max: 2.0, default: 0.8 }
    top_p: { min: 0.0, max: 1.0, default: 0.9 }
    top_k: { min: 1, max: 100, default: 40 }
    max_tokens: { min: 1, max: 32768, default: 2048 }
    repeat_penalty: { min: 0.0, max: 2.0, default: 1.1 }
    seed: { min: 0, max: 2^32, default: null }
  }
}

interface CompatibilityResult {
  compatible: boolean
  confidence: 'high' | 'medium' | 'low' | 'none'
  reasons: string[] // Why compatible or not
  warnings: string[] // Potential issues
  recommendations: string[] // Better alternatives
  estimated_ram_usage: number // NEW: Estimated RAM usage
}
```

---

**TEAM-406 Competitive Research Complete**  
**Next:** Hand off to TEAM-407 with these requirements

---

**TEAM-406 - Competitive Research v1.0**  
**Next Document:** `TEAM_406_HUGGINGFACE_FILTER_STRATEGY.md`

---

## 🔍 HuggingFace Marketplace Filter Strategy

### Core Principle: "If We Don't Support It, It Doesn't Exist"

**Philosophy:** Only show models that rbee workers can actually run. Filter out incompatible models BEFORE they reach the user.

**Why This Matters:**
- ✅ **Fool-proof UX:** Users can't install incompatible models
- ✅ **No confusion:** Unsupported formats/architectures never appear
- ✅ **Competitive advantage:** LM Studio has basic keyword search only
- ✅ **Performance:** Pre-filtered results = faster browsing

---

### Filter Criteria (Server-Side)

#### 1. Format Filter (CRITICAL)

**Current State (Phase 1):**
```typescript
// ONLY show SafeTensors models
filter: {
  format: ['safetensors']  // GGUF not supported yet
}
```

**Future State (Phase 2 - After GGUF Support):**
```typescript
// Show both SafeTensors and GGUF
filter: {
  format: ['safetensors', 'gguf']
}
```

**Implementation:**
- Check HuggingFace file list for `.safetensors` files
- Future: Check for `.gguf` files
- **HIDE models with only unsupported formats** (PyTorch, TensorFlow, etc.)

#### 2. Architecture Filter

**Supported Architectures (Current - Phase 1):**
```typescript
const SUPPORTED_ARCHITECTURES = [
  'llama',      // ✅ Tested
  'mistral',    // ⚠️ Code ready
  'phi',        // ⚠️ Code ready
  'qwen',       // ⚠️ Code ready
  'qwen2',      // ⚠️ Code ready
]
```

**Supported Architectures (Future - Phase 2):**
```typescript
const SUPPORTED_ARCHITECTURES = [
  // Phase 1 (current)
  'llama', 'mistral', 'phi', 'qwen', 'qwen2',
  
  // Phase 2 (expansion)
  'gemma', 'gemma2', 'gemma3',
  'deepseek', 'deepseek2',
  'falcon',
  'starcoder', 'starcoder2',
  'yi',
  'granite',
  
  // Phase 3 (advanced)
  'mixtral',
  'qwen2_moe', 'qwen3_moe',
  'phi3',
  'olmo', 'olmo2',
]
```

**Implementation:**
- Extract architecture from HuggingFace `config.json` → `model_type` or `architectures` field
- **HIDE models with unsupported architectures**
- Show "Coming Soon" badge for architectures in development

#### 3. Size Filter (User-Facing)

**Allow users to filter by model size:**
```typescript
const SIZE_FILTERS = [
  { label: 'Tiny (≤1B)', min: 0, max: 1_000_000_000 },
  { label: 'Small (1B-4B)', min: 1_000_000_000, max: 4_000_000_000 },
  { label: 'Medium (4B-9B)', min: 4_000_000_000, max: 9_000_000_000 },
  { label: 'Large (9B-20B)', min: 9_000_000_000, max: 20_000_000_000 },
  { label: 'XL (20B-70B)', min: 20_000_000_000, max: 70_000_000_000 },
  { label: 'XXL (>70B)', min: 70_000_000_000, max: Infinity },
]
```

**Extract from:**
- HuggingFace model card metadata
- Model name (e.g., "Llama-3.1-8B" → 8B)
- `config.json` parameter count

#### 4. Worker Compatibility Filter (rbee-Specific)

**Show only models compatible with installed workers:**
```typescript
interface WorkerCompatibilityFilter {
  // If user has CUDA worker installed
  show_cuda_compatible: boolean
  
  // If user has Metal worker installed
  show_metal_compatible: boolean
  
  // If user has CPU worker installed
  show_cpu_compatible: boolean
  
  // Show all if no workers installed (marketplace browsing)
  show_all_if_no_workers: boolean
}
```

**Logic:**
- Query installed workers from Keeper/Hive
- Filter models by architecture + format compatibility
- **HIDE models that won't run on user's hardware**

#### 5. Context Length Filter (User-Facing)

```typescript
const CONTEXT_LENGTH_FILTERS = [
  { label: 'Short (≤8K)', max: 8192 },
  { label: 'Medium (8K-32K)', min: 8192, max: 32768 },
  { label: 'Long (32K-128K)', min: 32768, max: 131072 },
  { label: 'Ultra (>128K)', min: 131072 },
]
```

**Extract from:**
- `config.json` → `max_position_embeddings` or `max_sequence_length`

---

### Model Detail Page - File List Filtering

**Problem:** HuggingFace models have 100+ files (checkpoints, configs, READMEs, etc.)

**Solution:** Show only relevant files by default, hide the rest

#### Relevant Files (Always Show)

```typescript
const RELEVANT_FILES = [
  // Model weights
  '*.safetensors',
  '*.gguf',
  
  // Configuration
  'config.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'generation_config.json',
  
  // Documentation
  'README.md',
  'MODEL_CARD.md',
  
  // License
  'LICENSE',
  'LICENSE.txt',
]
```

#### Hidden Files (Show More Button)

```typescript
const HIDDEN_FILES = [
  // PyTorch checkpoints (not supported)
  '*.bin',
  '*.pt',
  '*.pth',
  
  // TensorFlow (not supported)
  '*.h5',
  '*.pb',
  
  // Training artifacts
  'training_args.json',
  'trainer_state.json',
  'optimizer.pt',
  
  // Git metadata
  '.gitattributes',
  '.gitignore',
  
  // Other
  '*.msgpack',
  '*.json' (except relevant ones),
]
```

#### UI Implementation

```tsx
<FileList>
  {/* Always visible */}
  <FileSection title="Model Files">
    {relevantFiles.map(file => <FileItem key={file.name} file={file} />)}
  </FileSection>
  
  {/* Collapsed by default */}
  <Collapsible>
    <CollapsibleTrigger>
      Show {hiddenFiles.length} more files
    </CollapsibleTrigger>
    <CollapsibleContent>
      <FileSection title="Other Files">
        {hiddenFiles.map(file => <FileItem key={file.name} file={file} />)}
      </FileSection>
    </CollapsibleContent>
  </Collapsible>
</FileList>
```

---

### Default HuggingFace Search Parameters

**Marketplace SDK should use these defaults:**

```typescript
const DEFAULT_HF_SEARCH_PARAMS = {
  // Only show models, not datasets or spaces
  type: 'model',
  
  // Filter by supported formats
  filter: {
    // Phase 1: SafeTensors only
    library: ['transformers'],
    tags: ['safetensors'],
    
    // Phase 2: Add GGUF
    // tags: ['safetensors', 'gguf'],
  },
  
  // Sort by popularity (downloads)
  sort: 'downloads',
  direction: 'desc',
  
  // Pagination
  limit: 50,
  
  // Only show models with required files
  require_files: [
    'config.json',      // Required for architecture detection
    'tokenizer.json',   // Required for tokenization
  ],
}
```

---

### Advanced Filtering (Beat LM Studio)

**LM Studio doesn't have this - rbee can win here!**

#### Filter UI (Marketplace)

```tsx
<FilterPanel>
  <FilterSection title="Architecture">
    <Checkbox value="llama">Llama (1,234 models)</Checkbox>
    <Checkbox value="mistral">Mistral (567 models)</Checkbox>
    <Checkbox value="phi">Phi (234 models)</Checkbox>
    <Checkbox value="qwen">Qwen (345 models)</Checkbox>
    {/* ... */}
  </FilterSection>
  
  <FilterSection title="Size">
    <RadioGroup>
      <Radio value="tiny">Tiny (≤1B)</Radio>
      <Radio value="small">Small (1B-4B)</Radio>
      <Radio value="medium">Medium (4B-9B)</Radio>
      {/* ... */}
    </RadioGroup>
  </FilterSection>
  
  <FilterSection title="Format">
    <Checkbox value="safetensors">SafeTensors</Checkbox>
    <Checkbox value="gguf" disabled={!ggufSupported}>
      GGUF {!ggufSupported && '(Coming Soon)'}
    </Checkbox>
  </FilterSection>
  
  <FilterSection title="Context Length">
    <RadioGroup>
      <Radio value="short">Short (≤8K)</Radio>
      <Radio value="medium">Medium (8K-32K)</Radio>
      <Radio value="long">Long (32K-128K)</Radio>
      <Radio value="ultra">Ultra (>128K)</Radio>
    </RadioGroup>
  </FilterSection>
  
  <FilterSection title="Worker Compatibility">
    <Checkbox value="cuda">CUDA Compatible</Checkbox>
    <Checkbox value="metal">Metal Compatible</Checkbox>
    <Checkbox value="cpu">CPU Compatible</Checkbox>
  </FilterSection>
</FilterPanel>
```

---

### Implementation Phases

#### Phase 1: Basic Filtering (TEAM-407 to TEAM-409)
- ✅ Format filter (SafeTensors only)
- ✅ Architecture filter (4 architectures)
- ✅ Hide unsupported models
- ✅ File list filtering (show relevant only)

#### Phase 2: Advanced Filtering (TEAM-410)
- ✅ Size filter (parameter count)
- ✅ Context length filter
- ✅ Worker compatibility filter
- ✅ Advanced filter UI

#### Phase 3: Smart Filtering (Future)
- ✅ Quantization level filter
- ✅ License filter (commercial vs non-commercial)
- ✅ Language filter (English, Chinese, multilingual)
- ✅ Task filter (chat, code, reasoning, etc.)

---

### Success Metrics

**Phase 1 Complete:**
- ✅ 0% unsupported models shown to users
- ✅ File list shows only relevant files by default
- ✅ Faster browsing (pre-filtered results)

**Phase 2 Complete:**
- ✅ Advanced filtering UI implemented
- ✅ Beat LM Studio (they only have keyword search)
- ✅ Worker compatibility filter working

**Phase 3 Complete:**
- ✅ Smart filtering based on user's hardware
- ✅ Personalized model recommendations
- ✅ Best-in-class model discovery experience

---

**TEAM-406 - HuggingFace Filter Strategy**  
**Next:** Implement in marketplace-sdk (TEAM-408)
