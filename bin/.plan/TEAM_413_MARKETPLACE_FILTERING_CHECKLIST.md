# TEAM-413: Marketplace Filtering Implementation Checklist

**Created by:** TEAM-406  
**Date:** 2025-11-05  
**Mission:** Filter HuggingFace models so ONLY compatible models appear in marketplace  
**Goal:** Perfect SEO - only generate static pages for models that rbee workers can run  
**Status:** 🎯 READY TO START

---

## 🎯 Mission

**Problem:** Currently, `/app/models/[slug]/page.tsx` generates static pages for ALL HuggingFace models, including incompatible ones (PyTorch, TensorFlow, unsupported architectures).

**Solution:** Filter models at build time so ONLY compatible models get static pages generated.

**Impact:**
- ✅ Perfect SEO (only compatible models indexed)
- ✅ Fool-proof UX (users can't find incompatible models)
- ✅ Faster builds (fewer pages to generate)
- ✅ No confusion (unsupported models don't exist)

---

## 📋 Complete Checklist

### Phase 1: Define Compatibility Rules (2-3 hours)

- [ ] **Task 1.1:** Create `/frontend/packages/marketplace-node/src/compatibility/constants.ts`
  - [ ] Define `SUPPORTED_ARCHITECTURES` (llama, mistral, phi, qwen, qwen2)
  - [ ] Define `SUPPORTED_FORMATS` (safetensors only in Phase 1)
  - [ ] Define `REQUIRED_FILES` (config.json, tokenizer.json)
  - [ ] Define `WEIGHT_FILE_EXTENSIONS` (safetensors, gguf, pytorch, tensorflow)
  - [ ] Define `RELEVANT_FILE_PATTERNS` (show by default)
  - [ ] Define `HIDDEN_FILE_PATTERNS` (show in "Show More")
  - [ ] Export TypeScript types

- [ ] **Task 1.2:** Create `/frontend/packages/marketplace-node/src/compatibility/checker.ts`
  - [ ] Implement `checkModelCompatibility()` function
  - [ ] Implement `detectArchitecture()` from tags/model ID
  - [ ] Implement `checkRequiredFiles()` function
  - [ ] Implement `checkSupportedFormat()` function
  - [ ] Implement `filterRelevantFiles()` function
  - [ ] Add JSDoc comments to all functions
  - [ ] Export `CompatibilityResult` type

- [ ] **Task 1.3:** Update `/frontend/packages/marketplace-node/src/index.ts`
  - [ ] Export compatibility checker functions
  - [ ] Export compatibility constants
  - [ ] Export TypeScript types

### Phase 2: Filter HuggingFace API Calls (2-3 hours)

- [ ] **Task 2.1:** Update `fetchModels()` in `/frontend/packages/marketplace-node/src/huggingface.ts`
  - [ ] Add `onlyCompatible` parameter (default: true)
  - [ ] Add HuggingFace API filters (library: transformers, tags: safetensors)
  - [ ] Add client-side compatibility filtering
  - [ ] Test backward compatibility

- [ ] **Task 2.2:** Update `fetchModel()` in `/frontend/packages/marketplace-node/src/huggingface.ts`
  - [ ] Add compatibility check
  - [ ] Throw error if model incompatible
  - [ ] Include reasons in error message

### Phase 3: Filter Static Page Generation (1-2 hours)

- [ ] **Task 3.1:** Update `/frontend/apps/marketplace/app/models/[slug]/page.tsx`
  - [ ] Update `generateStaticParams()` to use `onlyCompatible: true`
  - [ ] Add double-check compatibility filter
  - [ ] Log count of compatible models
  - [ ] Warn about skipped incompatible models

- [ ] **Task 3.2:** Update model page to show compatibility info
  - [ ] Check compatibility in page component
  - [ ] Filter file list (relevant vs hidden)
  - [ ] Pass compatibility info to template
  - [ ] Handle incompatible models gracefully

- [ ] **Task 3.3:** Update `/frontend/apps/marketplace/app/models/page.tsx`
  - [ ] Use `onlyCompatible: true` parameter
  - [ ] Log count of compatible models
  - [ ] Only show compatible models in grid

### Phase 4: Update UI Components (3-4 hours)

- [ ] **Task 4.1:** Create `/frontend/packages/rbee-ui/src/components/CompatibilityBadge.tsx`
  - [ ] Create badge component (compatible/warning/incompatible)
  - [ ] Add lucide-react icons
  - [ ] Support size variants (sm, md, lg)
  - [ ] Export from rbee-ui

- [ ] **Task 4.2:** Create `/frontend/packages/rbee-ui/src/components/ModelFileList.tsx`
  - [ ] Show relevant files by default
  - [ ] Add "Show More" button for hidden files
  - [ ] Collapsible hidden files section
  - [ ] Format file sizes
  - [ ] Export from rbee-ui

- [ ] **Task 4.3:** Update `/frontend/packages/rbee-ui/src/components/ModelDetailPageTemplate.tsx`
  - [ ] Add compatibility badge to header
  - [ ] Add filtered file list
  - [ ] Show compatibility warnings (if any)
  - [ ] Ensure backward compatibility

### Phase 5: Testing & Verification (2-3 hours)

- [ ] **Task 5.1:** Create `/frontend/packages/marketplace-node/src/compatibility/checker.test.ts`
  - [ ] Test compatible model (llama + safetensors)
  - [ ] Test incompatible format (pytorch only)
  - [ ] Test unsupported architecture (bert)
  - [ ] Test missing required files
  - [ ] Test file filtering

- [ ] **Task 5.2:** Manual Testing
  - [ ] Run `npm run build` in marketplace app
  - [ ] Verify only compatible models get static pages
  - [ ] Check build logs for skipped models
  - [ ] Test model detail pages
  - [ ] Test file list "Show More" button
  - [ ] Test compatibility badges

- [ ] **Task 5.3:** Verification
  - [ ] Count generated static pages (should be < 1000)
  - [ ] Verify no PyTorch-only models in pages
  - [ ] Verify no unsupported architectures in pages
  - [ ] Check SEO metadata (only compatible models)
  - [ ] Test 404 for incompatible model slugs

---

## 📊 Effort Estimates

| Phase | Tasks | Effort | Priority |
|-------|-------|--------|----------|
| Phase 1: Compatibility Rules | 3 | 2-3 hours | 🔥 CRITICAL |
| Phase 2: API Filtering | 2 | 2-3 hours | 🔥 CRITICAL |
| Phase 3: Static Generation | 3 | 1-2 hours | 🔥 CRITICAL |
| Phase 4: UI Components | 3 | 3-4 hours | 🎯 HIGH |
| Phase 5: Testing | 3 | 2-3 hours | 🎯 HIGH |

**Total Effort:** 10-15 hours

---

## ✅ Success Criteria

### Phase 1 Complete:
- ✅ Compatibility constants defined
- ✅ Compatibility checker implemented
- ✅ All functions exported from marketplace-node

### Phase 2 Complete:
- ✅ `fetchModels()` filters by compatibility
- ✅ `fetchModel()` throws on incompatible models
- ✅ HuggingFace API filters applied

### Phase 3 Complete:
- ✅ `generateStaticParams()` only returns compatible models
- ✅ Build logs show count of compatible models
- ✅ No static pages for incompatible models

### Phase 4 Complete:
- ✅ Compatibility badge shown on model pages
- ✅ File list filtered (relevant vs hidden)
- ✅ "Show More" button works
- ✅ Compatibility warnings shown

### Phase 5 Complete:
- ✅ All tests passing
- ✅ Manual testing complete
- ✅ Build generates < 1000 pages (only compatible)
- ✅ No incompatible models accessible

---

## 🚨 Critical Files

### New Files (5):
1. `/frontend/packages/marketplace-node/src/compatibility/constants.ts`
2. `/frontend/packages/marketplace-node/src/compatibility/checker.ts`
3. `/frontend/packages/marketplace-node/src/compatibility/checker.test.ts`
4. `/frontend/packages/rbee-ui/src/components/CompatibilityBadge.tsx`
5. `/frontend/packages/rbee-ui/src/components/ModelFileList.tsx`

### Modified Files (4):
1. `/frontend/packages/marketplace-node/src/index.ts`
2. `/frontend/packages/marketplace-node/src/huggingface.ts`
3. `/frontend/apps/marketplace/app/models/[slug]/page.tsx`
4. `/frontend/packages/rbee-ui/src/components/ModelDetailPageTemplate.tsx`

---

## 📝 Implementation Notes

### Phase 1 Architecture Detection

**Current:** Detect from tags and model ID  
**Future (Phase 2):** Fetch `config.json` and check `model_type` or `architectures` field

**Why not now:** Requires additional API call per model (slow during build)

### Phase 1 Format Support

**Current:** SafeTensors only  
**Future (After GGUF support in workers):** Add GGUF

**Update required:** Change `SUPPORTED_FORMATS` constant

### File Filtering Strategy

**Relevant files (always show):**
- Model weights (`.safetensors`, `.gguf`)
- Configuration files (`config.json`, `tokenizer.json`)
- Documentation (`README.md`)
- License files

**Hidden files (show in "Show More"):**
- PyTorch/TensorFlow files (not supported)
- Training artifacts
- Git metadata

---

## 🎯 Next Steps After Complete

### Phase 2 (After GGUF Support in Workers):
- [ ] Add `'gguf'` to `SUPPORTED_FORMATS`
- [ ] Update HuggingFace API filter to include GGUF tag
- [ ] Test with GGUF models
- [ ] Update documentation

### Phase 3 (Architecture Expansion):
- [ ] Add new architectures to `SUPPORTED_ARCHITECTURES`
- [ ] Test with new architecture models
- [ ] Update compatibility badge logic

### Phase 4 (Advanced Filtering):
- [ ] Add size filter (parameter count)
- [ ] Add context length filter
- [ ] Add worker compatibility filter
- [ ] Add advanced filter UI

---

**TEAM-413 - Marketplace Filtering Checklist**  
**Total Effort:** 10-15 hours  
**Next:** Start with Phase 1 (Compatibility Rules)
