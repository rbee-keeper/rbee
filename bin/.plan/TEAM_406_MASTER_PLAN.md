# TEAM-406: Worker-Model Compatibility Matrix Implementation

**Created:** 2025-11-05  
**Team:** TEAM-406  
**Status:** 🎯 PLANNING PHASE  
**Engineering Rules:** ✅ READ (v2.0 - Rule Zero: Breaking Changes > Entropy)

---

## 🎯 Mission Statement

**Implement a complete worker-model compatibility matrix that:**
1. Shows which rbee workers can run which models
2. Integrates into marketplace pipeline (Next.js SSG + Tauri app)
3. Extends marketplace-sdk with worker catalog data
4. Matches or exceeds Ollama/LM Studio compatibility standards
5. Results in a production-ready compatibility chart for users

---

## 📋 What You Told Me (Raw Requirements)

### Core Requirements
1. **Compatibility Matrix** between:
   - Models in `/home/vince/Projects/llama-orch/frontend/apps/marketplace/app/models/[slug]/page.tsx`
   - Models in `/home/vince/Projects/llama-orch/frontend/apps/marketplace/app/models/page.tsx`
   - Workers in `/home/vince/Projects/llama-orch/bin/80-hono-worker-catalog/src/data.ts`

2. **marketplace-sdk Integration**:
   - Path: `/home/vince/Projects/llama-orch/bin/99_shared_crates/marketplace-sdk/src/lib.rs`
   - Purpose: Data layer for BOTH Tauri app AND Next.js app (SSG)
   - Needs: rbee workers entry

3. **marketplace-node Integration**:
   - Path: `/home/vince/Projects/llama-orch/frontend/packages/marketplace-node`
   - Needs: rbee workers entry

4. **Follow Complete Code Flow**:
   - Ensure NO code files are missed
   - Track data from worker catalog → SDK → UI

5. **artifacts-contract Update**:
   - Path: `/home/vince/Projects/llama-orch/bin/97_contracts/artifacts-contract/src/worker.rs`
   - Status: Outdated but probably needed

6. **Fix Documentation Warnings**:
   - Clean up all Rust doc warnings before implementation

7. **Competitive Analysis**:
   - Research Ollama compatibility objectives
   - Research LM Studio compatibility objectives
   - Define ideal LLM worker compatibility chart
   - Stay competitive with industry standards

---

## 🏗️ Architecture Context (Current State)

### Current Worker Catalog
**Location:** `bin/80-hono-worker-catalog/src/data.ts`

**Workers Available:**
- `llm-worker-rbee-cpu` (Linux, macOS, Windows)
- `llm-worker-rbee-cuda` (Linux, Windows)
- `llm-worker-rbee-metal` (macOS)
- `sd-worker-rbee-cpu` (Linux, macOS, Windows)
- `sd-worker-rbee-cuda` (Linux, Windows)
- `sd-worker-rbee-metal` (macOS)

**Current Capabilities:**
- Supported formats: `["gguf", "safetensors"]`
- Max context: 32,768 tokens
- Streaming: ✅ Yes
- Batching: ❌ No

### Current Model Support (llm-worker-rbee)
**Location:** `bin/30_llm_worker_rbee/docs/MODEL_SUPPORT.md`

**Architectures:**
- ✅ Llama (TinyLlama-1.1B tested on CPU/Metal/CUDA)
- ⚠️ Mistral (code ready, needs SafeTensors)
- ⚠️ Phi (code ready, needs SafeTensors)
- ⚠️ Qwen (code ready, needs SafeTensors)

**Format Support:**
- ✅ SafeTensors (primary)
- ❌ GGUF (downloaded but not supported yet)

**Backend Status:**
- ✅ CPU: Working
- ✅ Metal: Working (with Candle fork fix)
- ✅ CUDA: Working (with Candle fork fix)

### Current Marketplace Architecture
**Next.js App:** `frontend/apps/marketplace/`
- SSG pages for 1000+ models
- HuggingFace integration
- Model detail pages with slugified URLs

**marketplace-sdk:** `bin/99_shared_crates/marketplace-sdk/`
- Rust + WASM SDK
- Types defined (WorkerType, Platform)
- HuggingFace client (native Rust, non-WASM)
- ❌ Worker catalog client NOT implemented
- ❌ WASM NOT built

**marketplace-node:** `frontend/packages/marketplace-node/`
- Node.js wrapper for WASM
- ❌ Worker functions return empty arrays (TODOs)

**artifacts-contract:** `bin/97_contracts/artifacts-contract/src/worker.rs`
- Canonical WorkerType enum (Cpu, Cuda, Metal)
- Canonical Platform enum (Linux, MacOS, Windows)
- WorkerBinary struct
- ⚠️ May be outdated vs Hono catalog

---

## 🎯 Gap Analysis

### What's Missing

1. **Worker Catalog in marketplace-sdk**
   - No Rust client for worker catalog
   - No WASM bindings for workers
   - marketplace-node can't list workers

2. **Compatibility Matrix Data**
   - No mapping: worker → supported models
   - No mapping: model → compatible workers
   - No format compatibility checks (GGUF vs SafeTensors)

3. **Model Metadata**
   - Models don't expose architecture (Llama, Mistral, etc.)
   - Models don't expose format (GGUF, SafeTensors)
   - Models don't expose quantization level

4. **Worker Capabilities**
   - Workers don't expose supported architectures
   - Workers don't expose format support details
   - No generation parameter compatibility info

5. **Competitive Parity**
   - Unknown: Ollama compatibility standards
   - Unknown: LM Studio compatibility standards
   - No benchmark for "ideal" compatibility chart

6. **Documentation**
   - Rust doc warnings need fixing
   - No compatibility matrix documentation
   - No user-facing compatibility guide

---

## 📊 Competitive Research Needed

### Ollama Compatibility (Research Phase)
**Questions to Answer:**
1. Which model architectures does Ollama support?
2. Which formats does Ollama support? (GGUF, SafeTensors, etc.)
3. What quantization levels does Ollama support?
4. What generation parameters does Ollama expose?
5. How does Ollama communicate compatibility to users?

**Sources:**
- Ollama GitHub repo
- Ollama documentation
- Ollama model library

### LM Studio Compatibility (Research Phase)
**Questions to Answer:**
1. Which model architectures does LM Studio support?
2. Which formats does LM Studio support?
3. What quantization levels does LM Studio support?
4. What generation parameters does LM Studio expose?
5. How does LM Studio communicate compatibility to users?

**Sources:**
- LM Studio website
- LM Studio documentation
- LM Studio model compatibility matrix

### Industry Standards (Research Phase)
**Questions to Answer:**
1. What's the minimum viable compatibility matrix?
2. What generation parameters are table stakes? (temperature, top_p, top_k, etc.)
3. What model formats are industry standard?
4. What quantization levels are most common?

**Reference:**
- OpenAI API (temperature, top_p, frequency_penalty, etc.)
- Anthropic Claude API
- llama.cpp parameters
- Current rbee state: `bin/.specs/.docs/GENERATION_PARAMETERS_ANALYSIS.md`

---

## 🗺️ Implementation Phases (High-Level)

### Phase 0: Planning & Research (TEAM-406)
**Duration:** 2-3 days  
**Deliverables:**
- ✅ This master plan
- 📋 Competitive research report (Ollama, LM Studio)
- 📋 Ideal compatibility matrix specification
- 📋 6 detailed phase checklists

### Phase 1: Fix Documentation & Contracts (TEAM-407)
**Duration:** 1 day  
**Deliverables:**
- Fix all Rust doc warnings
- Update artifacts-contract if needed
- Align worker.rs with Hono catalog types
- Ensure type consistency across codebase

### Phase 2: Worker Catalog in marketplace-sdk (TEAM-408)
**Duration:** 2-3 days  
**Deliverables:**
- Rust worker catalog client
- WASM bindings for workers
- marketplace-node worker functions
- Tests

### Phase 3: Compatibility Matrix Data Layer (TEAM-409)
**Duration:** 3-4 days  
**Deliverables:**
- Worker → model compatibility mappings
- Model metadata (architecture, format, quantization)
- Compatibility check functions
- Tests

### Phase 4: Next.js Integration (TEAM-410)
**Duration:** 2-3 days  
**Deliverables:**
- Compatibility matrix on model detail pages
- Worker recommendations per model
- SSG for all compatibility data
- SEO optimization

### Phase 5: Tauri Integration (TEAM-411)
**Duration:** 2-3 days  
**Deliverables:**
- Compatibility matrix in Keeper UI
- Worker selection based on compatibility
- Install flow with compatibility checks
- Tests

### Phase 6: Documentation & Launch (TEAM-412)
**Duration:** 1-2 days  
**Deliverables:**
- User-facing compatibility guide
- Developer documentation
- Migration guide (if breaking changes)
- Launch checklist

---

## 📁 Files to Track (Complete Code Flow)

### Rust Backend
- `bin/97_contracts/artifacts-contract/src/worker.rs` - Canonical types
- `bin/99_shared_crates/marketplace-sdk/src/lib.rs` - SDK entry point
- `bin/99_shared_crates/marketplace-sdk/src/types.rs` - Type definitions
- `bin/99_shared_crates/marketplace-sdk/Cargo.toml` - Dependencies
- `bin/80-hono-worker-catalog/src/data.ts` - Worker catalog data
- `bin/80-hono-worker-catalog/src/types.ts` - Worker types
- `bin/30_llm_worker_rbee/docs/MODEL_SUPPORT.md` - Model support matrix

### TypeScript/WASM
- `frontend/packages/marketplace-node/src/index.ts` - Node.js wrapper
- `frontend/packages/marketplace-node/package.json` - Dependencies
- `bin/99_shared_crates/marketplace-sdk/pkg/` - WASM output (needs build)

### Next.js Frontend
- `frontend/apps/marketplace/app/models/page.tsx` - Model list page
- `frontend/apps/marketplace/app/models/[slug]/page.tsx` - Model detail page
- `frontend/apps/marketplace/lib/huggingface.ts` - HuggingFace client
- `frontend/apps/marketplace/lib/slugify.ts` - URL helpers

### Tauri App
- `bin/00_rbee_keeper/ui/src/pages/MarketplacePage.tsx` - Marketplace page
- `bin/00_rbee_keeper/src/handlers/protocol.rs` - Protocol handler

### Documentation
- `bin/.plan/README.md` - Master plan index
- `bin/.plan/CHECKLIST_02_MARKETPLACE_SDK.md` - SDK checklist
- `bin/.plan/MARKETPLACE_SHARED_COMPONENTS_ARCHITECTURE.md` - Component design

---

## ⚠️ Critical Constraints (Engineering Rules)

### Rule Zero: Breaking Changes > Entropy
- ✅ Update existing functions, don't create `_v2()` versions
- ✅ Delete deprecated code immediately
- ✅ One way to do things, not 3 different APIs
- ✅ Compiler will find all call sites - fix them

### Code Quality
- ✅ Add TEAM-406 signatures to all new code
- ✅ Complete previous team's TODO list (CHECKLIST_02 Phase 2-6)
- ❌ NO background testing
- ❌ NO TODO markers in final code

### Documentation
- ✅ Update existing docs, don't create duplicates
- ✅ Max 2 pages for handoffs
- ❌ NO multiple .md files for one task

### Destructive Actions
- ✅ Delete dead code immediately
- ✅ Remove deprecated functions
- ✅ Break APIs if needed (pre-1.0)

---

## 🎯 Success Criteria

### Phase 0 (This Document)
- [x] Master plan created
- [ ] Competitive research complete
- [ ] Ideal compatibility spec defined
- [ ] 6 phase checklists created

### Overall Success
- [ ] Users can see which workers run which models
- [ ] Compatibility matrix on every model detail page
- [ ] Worker selection in Keeper based on compatibility
- [ ] Matches or exceeds Ollama/LM Studio standards
- [ ] All Rust doc warnings fixed
- [ ] All tests passing
- [ ] Documentation complete

---

## 📚 Next Steps

### Immediate (TEAM-406)
1. Research Ollama compatibility standards
2. Research LM Studio compatibility standards
3. Define ideal compatibility matrix spec
4. Create 6 detailed phase checklists

### After Planning
1. TEAM-407: Fix docs & contracts (Phase 1)
2. TEAM-408: Worker catalog SDK (Phase 2)
3. TEAM-409: Compatibility data layer (Phase 3)
4. TEAM-410: Next.js integration (Phase 4)
5. TEAM-411: Tauri integration (Phase 5)
6. TEAM-412: Documentation & launch (Phase 6)

---

**TEAM-406 - Master Plan v1.0**  
**Next Document:** `TEAM_406_COMPETITIVE_RESEARCH.md`
