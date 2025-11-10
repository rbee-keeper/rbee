# CivitAI Types - Rule Zero Cleanup Progress

**TEAM-463: Thorough Type Deduplication**  
**Date:** 2025-11-10  
**Status:** 🚧 IN PROGRESS

## Objective

Apply Rule Zero to CivitAI types - eliminate ALL duplicates and establish `artifacts-contract` as the single source of truth.

## ✅ Completed Steps

### 1. Audit Complete
Found **8 duplicate type definitions** across 3 locations:
- `marketplace-sdk/src/wasm_civitai.rs` (6 structs)
- `marketplace-sdk/src/civitai.rs` (12 structs)
- `marketplace-node/src/civitai.ts` (3 interfaces)

### 2. Contract Types Created ✅
Created canonical types in `artifacts-contract/src/model/civitai.rs`:
- `CivitaiModel` - Main model type
- `CivitaiModelVersion` - Version info
- `CivitaiStats` - Statistics
- `CivitaiCreator` - Author info
- `CivitaiFile` - File info
- `CivitaiImage` - Preview images

**Features:**
- ✅ WASM-compatible (`#[cfg_attr(target_arch = "wasm32", derive(Tsify))]`)
- ✅ Specta support for Tauri (`#[cfg_attr(..., derive(specta::Type))]`)
- ✅ Proper serde attributes (`rename_all = "camelCase"`)
- ✅ Exported from `artifacts-contract/src/lib.rs`
- ✅ Compiles cleanly

## ✅ Completed Steps (Continued)

### 3. Deleted Duplicate WASM Types ✅
- ✅ Deleted `marketplace-sdk/src/wasm_civitai.rs` (complete duplicate)
- ✅ Removed wasm_civitai module references from `lib.rs`
- ✅ Updated exports to use contract types

### 4. Updated marketplace-sdk ✅
- ✅ Imported `CivitaiStats`, `CivitaiCreator` from `artifacts-contract`
- ✅ Kept internal API response types for parsing:
  - `CivitaiModelResponse` (pub(crate))
  - `CivitaiModelVersionResponse` (pub(crate))
  - `CivitaiFileResponse` (pub(crate))
  - `CivitaiImageResponse` (pub(crate))
- ✅ Deleted duplicate type definitions
- ✅ Added public API methods:
  - `get_marketplace_model()` - returns `Model`
  - `get_compatible_marketplace_models()` - returns `Vec<Model>`
- ✅ Made internal methods `pub(crate)`:
  - `get_model()` - returns internal response type
  - `to_marketplace_model()` - converts to Model
- ✅ Updated tauri commands to use new public API
- ✅ Verified compilation (keeper builds successfully)

### 5. Deleted marketplace-node Duplicates ✅
- ✅ Deleted `CivitAIModel` interface (68 lines)
- ✅ Deleted `CivitAIModelVersion` interface (28 lines)
- ✅ Imported from WASM-generated types (`../wasm/marketplace_sdk`)
- ✅ Added backward-compatible type aliases
- ✅ Kept `CivitAISearchResponse` (API-specific pagination wrapper)

### 6. Updated All Imports ✅
- ✅ marketplace-node uses WASM-generated types
- ✅ Added optional `createdAt`/`updatedAt` to contract types
- ✅ Updated TypeScript conversion to handle null → undefined
- ✅ All code uses contract types as source of truth

### 7. Verified Compilation ✅
- ✅ `cargo check -p artifacts-contract` - PASS
- ✅ `cargo check -p marketplace-sdk` - PASS
- ✅ `cargo check --bin rbee-keeper` - PASS
- ✅ `cd marketplace-node && npx tsc --noEmit` - PASS

## 🎉 100% COMPLETE!

All CivitAI types have been successfully deduplicated and moved to artifacts-contract as the single source of truth.

## Type Flow (Target Architecture)

```
CivitAI API (raw JSON)
  ↓ parse into
CivitaiModelResponse (internal SDK type for parsing)
  ↓ convert to
artifacts-contract::CivitaiModel (canonical type)
  ↓ re-exported by
marketplace-sdk
  ↓ generates WASM types
marketplace-node (imports from WASM)
  ↓ used by
UI components (TypeScript)
```

## Files to Modify

**Delete:**
- ❌ `marketplace-sdk/src/wasm_civitai.rs` (complete duplicate)
- ❌ `marketplace-node/src/civitai.ts` interfaces (partial duplicate)

**Update:**
- 🔧 `marketplace-sdk/src/civitai.rs` - use contract types
- 🔧 `marketplace-sdk/src/lib.rs` - export contract types
- 🔧 `marketplace-node/src/civitai.ts` - import from WASM
- 🔧 `marketplace-node/src/index.ts` - use WASM types

## Verification Checklist

- [ ] Contract types compile
- [ ] WASM types generate correctly
- [ ] marketplace-sdk compiles
- [ ] marketplace-node TypeScript compiles
- [ ] rbee-keeper builds
- [ ] No duplicate type definitions remain
- [ ] Documentation complete

## Next Action

Continue with Step 3: Delete `marketplace-sdk/src/wasm_civitai.rs`
