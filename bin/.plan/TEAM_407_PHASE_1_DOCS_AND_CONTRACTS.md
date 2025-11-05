# TEAM-407: Phase 1 - Fix Documentation & Contracts

**Created:** 2025-11-05  
**Team:** TEAM-407  
**Duration:** 1 day  
**Status:** ⏳ WAITING (blocked by TEAM-406 research)  
**Dependencies:** TEAM-406 competitive research complete

---

## 🎯 Mission

Fix all Rust documentation warnings and ensure type consistency across the codebase before implementing compatibility matrix.

---

## ✅ Checklist

### Task 1.1: Fix Rust Doc Warnings
- [ ] Run `cargo doc --workspace` and capture all warnings
- [ ] Create list of files with doc warnings
- [ ] Fix missing doc comments
- [ ] Fix broken doc links
- [ ] Fix incorrect doc examples
- [ ] Re-run `cargo doc --workspace` until clean
- [ ] Commit: "TEAM-407: Fix all Rust doc warnings"

**Files to Check:**
- `bin/99_shared_crates/marketplace-sdk/src/lib.rs`
- `bin/97_contracts/artifacts-contract/src/worker.rs`
- `bin/80-hono-worker-catalog/src/` (if Rust files exist)
- All marketplace-sdk modules

**Acceptance:**
- ✅ `cargo doc --workspace` produces ZERO warnings
- ✅ All public items have doc comments
- ✅ All doc links resolve correctly

---

### Task 1.2: Audit artifacts-contract/worker.rs
- [ ] Read current `bin/97_contracts/artifacts-contract/src/worker.rs`
- [ ] Read Hono catalog `bin/80-hono-worker-catalog/src/types.ts`
- [ ] Compare WorkerType enums (Rust vs TypeScript)
- [ ] Compare Platform enums (Rust vs TypeScript)
- [ ] Compare WorkerCatalogEntry vs WorkerBinary
- [ ] Document discrepancies in `TEAM_407_TYPE_AUDIT.md`
- [ ] Decide: Update Rust? Update TypeScript? Both?

**Questions to Answer:**
1. Is WorkerType enum consistent? (Cpu, Cuda, Metal)
2. Is Platform enum consistent? (Linux, MacOS, Windows)
3. Are there fields in TypeScript not in Rust?
4. Are there fields in Rust not in TypeScript?
5. Which is the source of truth?

**Deliverable:**
- `TEAM_407_TYPE_AUDIT.md` with findings and recommendations

---

### Task 1.3: Align Worker Types (If Needed)
- [ ] Based on audit, update artifacts-contract/worker.rs OR
- [ ] Update Hono catalog types.ts OR
- [ ] Update both to match
- [ ] Ensure tsify annotations are correct
- [ ] Ensure serde annotations are correct
- [ ] Run `cargo check -p artifacts-contract`
- [ ] Commit: "TEAM-407: Align worker types across Rust/TypeScript"

**Acceptance:**
- ✅ WorkerType enum matches across Rust/TypeScript
- ✅ Platform enum matches across Rust/TypeScript
- ✅ WorkerBinary/WorkerCatalogEntry fields aligned
- ✅ No compilation errors

---

### Task 1.4: Add Missing Worker Capabilities
- [ ] Review ideal compatibility spec from TEAM-406
- [ ] Check if WorkerBinary has all needed fields:
  - [ ] `supported_architectures: Vec<String>`
  - [ ] `supported_formats: Vec<String>`
  - [ ] `max_context_length: u32`
  - [ ] `supports_streaming: bool`
  - [ ] `supports_batching: bool`
  - [ ] `supported_parameters: HashMap<String, ParameterSpec>` (if needed)
- [ ] Add missing fields to artifacts-contract/worker.rs
- [ ] Update tsify annotations
- [ ] Update Hono catalog data.ts with new fields
- [ ] Run `cargo check -p artifacts-contract`
- [ ] Commit: "TEAM-407: Add worker capability fields"

**Acceptance:**
- ✅ WorkerBinary has all fields needed for compatibility checks
- ✅ Hono catalog data populated with capability data
- ✅ Types compile and generate correct TypeScript

---

### Task 1.5: Add Model Metadata Types
- [ ] Create `ModelMetadata` struct in artifacts-contract
- [ ] Add fields:
  - [ ] `architecture: ModelArchitecture` (enum)
  - [ ] `format: ModelFormat` (enum)
  - [ ] `quantization: Option<Quantization>` (enum)
  - [ ] `parameters: String` (e.g., "7B")
  - [ ] `size_bytes: u64`
  - [ ] `max_context_length: u32`
- [ ] Add tsify annotations
- [ ] Export from artifacts-contract
- [ ] Run `cargo check -p artifacts-contract`
- [ ] Commit: "TEAM-407: Add ModelMetadata types"

**Enums to Create:**
```rust
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "lowercase")]
pub enum ModelArchitecture {
    Llama,
    Mistral,
    Phi,
    Qwen,
    Gemma,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "lowercase")]
pub enum ModelFormat {
    SafeTensors,
    Gguf,
    Pytorch,
}

#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "lowercase")]
pub enum Quantization {
    Fp16,
    Fp32,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
}
```

**Acceptance:**
- ✅ ModelMetadata struct defined
- ✅ Enums defined with tsify
- ✅ Types compile
- ✅ TypeScript types generated

---

### Task 1.6: Update marketplace-sdk Types
- [ ] Import ModelMetadata from artifacts-contract
- [ ] Import updated WorkerBinary from artifacts-contract
- [ ] Update marketplace-sdk/src/types.rs
- [ ] Re-export new types
- [ ] Run `cargo check -p marketplace-sdk`
- [ ] Commit: "TEAM-407: Update marketplace-sdk types"

**Acceptance:**
- ✅ marketplace-sdk exports ModelMetadata
- ✅ marketplace-sdk exports updated WorkerBinary
- ✅ No compilation errors

---

### Task 1.7: Verification
- [ ] Run `cargo doc --workspace` - ZERO warnings
- [ ] Run `cargo check --workspace` - ZERO errors
- [ ] Run `cargo test -p artifacts-contract` - ALL PASS
- [ ] Run `cargo test -p marketplace-sdk` - ALL PASS
- [ ] Review all changes for TEAM-407 signatures
- [ ] Create handoff document (max 2 pages)

**Handoff Document Contents:**
- What was fixed (doc warnings count)
- What was aligned (type discrepancies)
- What was added (new fields/types)
- Verification results
- Next team ready: TEAM-408

---

## 📁 Files Modified

### Expected Changes
- `bin/97_contracts/artifacts-contract/src/worker.rs` - Updated types
- `bin/97_contracts/artifacts-contract/src/model.rs` - NEW (ModelMetadata)
- `bin/97_contracts/artifacts-contract/src/lib.rs` - Re-exports
- `bin/99_shared_crates/marketplace-sdk/src/types.rs` - Updated imports
- `bin/99_shared_crates/marketplace-sdk/src/lib.rs` - Updated re-exports
- `bin/80-hono-worker-catalog/src/data.ts` - Updated with capabilities
- `bin/80-hono-worker-catalog/src/types.ts` - Aligned with Rust

### New Files
- `TEAM_407_TYPE_AUDIT.md` - Type consistency audit
- `TEAM_407_HANDOFF.md` - Handoff document

---

## ⚠️ Blockers & Dependencies

### Blocked By
- TEAM-406 competitive research (need ideal spec for capabilities)

### Blocks
- TEAM-408 (needs clean types to implement worker catalog)

---

## 🎯 Success Criteria

- [ ] ZERO Rust doc warnings
- [ ] Worker types consistent across Rust/TypeScript
- [ ] Model metadata types defined
- [ ] All capability fields present
- [ ] All tests passing
- [ ] Handoff document complete (≤2 pages)

---

## 📚 References

- Engineering Rules: `.windsurf/rules/engineering-rules.md`
- Current worker types: `bin/97_contracts/artifacts-contract/src/worker.rs`
- Hono catalog: `bin/80-hono-worker-catalog/src/types.ts`
- Model support: `bin/30_llm_worker_rbee/docs/MODEL_SUPPORT.md`

---

**TEAM-407 - Phase 1 Checklist v1.0**  
**Next Phase:** TEAM-408 (Worker Catalog SDK)
