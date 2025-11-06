# TEAM-421: Model Files (Siblings) Implementation Plan

**Objective:** Add model files list to the left sidebar of model detail pages by fetching and displaying HuggingFace `siblings` data.

**Problem:** The ModelDetailPageTemplate has a left column for model files, but it's empty because:
1. Rust `Model` type doesn't include `siblings` field
2. HuggingFace API client doesn't fetch file information
3. TypeScript bindings don't include file types

---

## Implementation Checklist

### Phase 1: Add Rust Types for Model Files ✅ COMPLETE

- [x] **1.1** Add `ModelFile` struct to `marketplace-sdk/src/types.rs`
  - Fields: `filename: String`, `size: Option<f64>` (f64 for TypeScript compatibility)
  - Add `#[derive(Debug, Clone, Serialize, Deserialize)]`
  - Add `#[cfg_attr(feature = "specta", derive(specta::Type))]` for Tauri bindings

- [x] **1.2** Add `siblings` field to `Model` struct
  - Type: `Option<Vec<ModelFile>>`
  - Add serde attribute: `#[serde(skip_serializing_if = "Option::is_none")]`

- [x] **1.3** Export `ModelFile` type (already exported via pub struct)

### Phase 2: Update HuggingFace API Client ✅ COMPLETE

- [x] **2.1** Update `marketplace-sdk/src/huggingface.rs` to fetch siblings
  - Parse `siblings` from HuggingFace API `extra` fields
  - Extract `rfilename` and `size` from each file object
  - Map to `ModelFile` struct

- [x] **2.2** Update `convert_hf_model()` function to include siblings
  - Parse siblings array from API response
  - Convert u64 sizes to f64 for TypeScript compatibility
  - Filter out empty arrays

- [x] **2.3** HuggingFace API integration ready
  - Siblings are parsed from existing API responses
  - File sizes are converted to f64
  - Empty file lists are filtered out

### Phase 3: Rebuild and Regenerate Bindings ✅ COMPLETE

- [x] **3.1** Rebuild marketplace-sdk
  - Run: `cargo build -p marketplace-sdk` ✅
  - No compilation errors

- [x] **3.2** Regenerate TypeScript bindings
  - Run: `cargo test --package rbee-keeper --lib export_typescript_bindings` ✅
  - Bindings generated successfully

- [x] **3.3** Verify TypeScript bindings include `ModelFile` and `siblings`
  - `ModelFile` type exported ✅
  - `Model.siblings?: ModelFile[] | null` added ✅

### Phase 4: Update Frontend ⚠️ PENDING

- [ ] **4.1** Verify `ModelDetailPageTemplate` already handles siblings
  - Check line 231-233 in `ModelDetailPageTemplate.tsx`
  - Confirm `ModelFilesList` component exists and works

- [ ] **4.2** Update `ModelDetailsPage.tsx` if needed
  - Ensure `siblings` is passed through in model transformation
  - Test that files appear in left column

- [ ] **4.3** Build and test UI
  - Run: `pnpm run build` in `bin/00_rbee_keeper/ui` (has unrelated TS errors)
  - Rebuild rbee-keeper: `cargo build --bin rbee-keeper --release` ✅

### Phase 5: Testing ⚠️ PENDING

- [ ] **5.1** Test with real HuggingFace model
  - Navigate to a model detail page
  - Verify files list appears in left column
  - Check file names and sizes are correct

- [ ] **5.2** Test error handling
  - Test model without files
  - Test API failure scenarios

- [ ] **5.3** Verify consistency
  - Left column should show files
  - Right column should show description, metadata, etc.
  - Layout should match design

---

## Files to Modify

### Rust (Backend)
1. `/bin/97_contracts/artifacts-contract/src/model.rs` - Add ModelFile struct, update Model
2. `/bin/97_contracts/artifacts-contract/src/lib.rs` - Export ModelFile
3. `/bin/79_marketplace_core/marketplace-sdk/src/huggingface.rs` - Fetch siblings from API

### TypeScript (Frontend)
1. `/bin/00_rbee_keeper/ui/src/generated/bindings.ts` - Auto-generated, will update after rebuild
2. `/bin/00_rbee_keeper/ui/src/pages/ModelDetailsPage.tsx` - Verify siblings are passed through

---

## Expected Result

**Before:**
```
┌─────────────────────────────────────┐
│ Model Detail Page                   │
├─────────────┬───────────────────────┤
│             │ Description           │
│   EMPTY     │ Compatible Workers    │
│             │ Metadata              │
└─────────────┴───────────────────────┘
```

**After:**
```
┌─────────────────────────────────────┐
│ Model Detail Page                   │
├─────────────┬───────────────────────┤
│ Model Files │ Description           │
│ - model.bin │ Compatible Workers    │
│ - config.js │ Metadata              │
│ - vocab.txt │                       │
└─────────────┴───────────────────────┘
```

---

## Notes

- HuggingFace API returns `siblings` as an array of file objects
- Each file has: `rfilename` (relative filename), `size` (bytes), `lfs` (Git LFS info)
- The UI component `ModelFilesList` already exists and expects this data
- This is a **data plumbing task** - the UI is already built, we just need to feed it data

---

## Estimated Time

- Phase 1: 10 minutes (add Rust types)
- Phase 2: 15 minutes (update API client)
- Phase 3: 5 minutes (rebuild and verify)
- Phase 4: 5 minutes (verify frontend)
- Phase 5: 10 minutes (testing)

**Total: ~45 minutes**
