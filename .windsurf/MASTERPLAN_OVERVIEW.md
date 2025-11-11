# Masterplan: Fix Filter Functionality Using Proper WASM Architecture

**Created**: 2025-11-11  
**Status**: Planning Complete  
**Estimated Time**: 4-6 hours

## Overview

Fix the HuggingFace model filtering by properly integrating the existing Rust/WASM SDK infrastructure instead of using hacky TypeScript fetch calls.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ marketplace-sdk (Rust)                                       │
│ - huggingface.rs: Core API client                           │
│ - types.rs: Model, ModelFile definitions                    │
│ - wasm_huggingface.rs: WASM bindings                        │
│ ❌ NEVER imported by frontend apps                          │
└─────────────────────────────────────────────────────────────┘
                          ↓ (compile to WASM)
┌─────────────────────────────────────────────────────────────┐
│ marketplace-node (TypeScript wrapper)                        │
│ - Wraps WASM for Node.js                                    │
│ - Used at BUILD TIME only                                   │
│ - Exports: listHuggingFaceModels(), etc.                    │
│ ✅ THIS is what frontend imports                            │
└─────────────────────────────────────────────────────────────┘
                          ↓ (BUILD TIME)
┌─────────────────────────────────────────────────────────────┐
│ Manifest Generation Script (generate-model-manifests.ts)    │
│ - Uses marketplace-node (NOT marketplace-sdk directly!)     │
│ - Parallel fetching with rate limiting (3 concurrent)       │
│ - Fetches models via WASM                                   │
│ - Filters by size/sort/license                              │
│ - Outputs: /public/manifests/*.json                         │
└─────────────────────────────────────────────────────────────┘
                          ↓ (outputs)
┌─────────────────────────────────────────────────────────────┐
│ Static JSON Manifests                                        │
│ - /public/manifests/hf-filter-small.json                    │
│ - /public/manifests/hf-filter-medium.json                   │
│ - Lightweight: {id, slug, name}                             │
└─────────────────────────────────────────────────────────────┘
                          ↓ (RUNTIME)
┌─────────────────────────────────────────────────────────────┐
│ Next.js Frontend                                             │
│ - Loads manifests from /public/manifests/                   │
│ - Enriches with SSG data                                    │
│ - Client-side filtering via URL params                      │
│ - Imports: @rbee/marketplace-node (types only at runtime)   │
└─────────────────────────────────────────────────────────────┘
                          ↓ (ALSO RUNTIME)
┌─────────────────────────────────────────────────────────────┐
│ Tauri App                                                    │
│ - Uses marketplace-sdk directly (native Rust)               │
│ - No WASM needed (native binary)                            │
│ - Calls same HuggingFaceClient                              │
│ - Imports Rust crate, not TypeScript wrapper                │
└─────────────────────────────────────────────────────────────┘
```

## Import Boundaries (CRITICAL)

```typescript
// ✅ ALLOWED: Frontend apps import marketplace-node
import { listHuggingFaceModels, type Model } from '@rbee/marketplace-node'

// ❌ FORBIDDEN: Frontend apps import marketplace-sdk
import { HuggingFaceClient } from '@rbee/marketplace-sdk'  // NEVER DO THIS!
```

**Why?**
- `marketplace-sdk` is Rust/WASM - only for Node.js build scripts
- Frontend runtime loads static manifests, doesn't call SDK
- Only `marketplace-node` should import `marketplace-sdk`
- Tauri uses native Rust SDK (not WASM, not marketplace-node)

## Phases

### Phase 1: Dependency Setup & Architecture Verification
**Time**: 30 minutes  
**File**: `PHASE_1_DEPENDENCIES.md`

- Add `@rbee/marketplace-node` to frontend dependencies
- Verify WASM builds correctly
- Verify Rust SDK types match TypeScript needs
- Document the Model/HFModel type mappings

### Phase 2: Rewrite Manifest Generation
**Time**: 2 hours  
**File**: `PHASE_2_MANIFEST_GENERATION.md`

- Delete hacky TypeScript fetch code
- Use `listHuggingFaceModels()` from WASM
- Implement proper size filtering using SDK data
- Add proper error handling
- Generate manifests with real data

### Phase 3: Fix Filter UI Integration
**Time**: 1-2 hours  
**File**: `PHASE_3_FILTER_UI_FIX.md`

- Debug why filter clicks don't update URL
- Fix the `handleFilterChange` callback chain
- Ensure `onChange` is properly passed through components
- Test that URL updates when filters are clicked

### Phase 4: End-to-End Testing
**Time**: 1 hour  
**File**: `PHASE_4_TESTING.md`

- Test single filter (Small/Medium/Large)
- Test multiple filters (Small + Most Likes)
- Test filter removal (click "All Sizes")
- Verify manifests load correctly
- Verify metadata displays
- Verify no infinite loops

### Phase 5: Tauri Integration
**Time**: 1 hour  
**File**: `PHASE_5_TAURI_INTEGRATION.md`

- Verify Tauri can use marketplace-sdk directly
- Add HuggingFace browsing to Tauri app
- Test that both web and desktop use same SDK
- Document the dual-use pattern

## Success Criteria

### Must Have ✅
- [ ] Manifests generated using marketplace-node WASM
- [ ] Small/Medium/Large filters show different data
- [ ] Filter clicks update URL correctly
- [ ] Multiple filters work together
- [ ] Metadata (downloads/likes/author) displays
- [ ] No infinite loops
- [ ] Tauri can use the SDK

### Nice to Have 🎯
- [ ] Manifest generation runs in CI/CD
- [ ] Types are fully type-safe (no `any`)
- [ ] Error states handled gracefully
- [ ] Loading states during manifest fetch
- [ ] Puppeteer tests for filter interactions

## Risk Mitigation

### Risk 1: WASM Doesn't Work in Build Script
**Mitigation**: Test marketplace-node in isolation first  
**Fallback**: Use Node.js fetch wrapper temporarily

### Risk 2: Filter Click Still Doesn't Work
**Mitigation**: Add debug logging at every step  
**Fallback**: Investigate if it's a Radix UI issue

### Risk 3: Manifests Too Large
**Mitigation**: Limit to 100 models per manifest  
**Fallback**: Use pagination/lazy loading

## Dependencies

### Required Packages
- `@rbee/marketplace-node` (already built) ✅
- `@rbee/marketplace-sdk` (Rust crate) ✅

### Development Tools
- `wasm-pack` (for WASM compilation) ✅
- `tsx` (for running TypeScript scripts) ✅
- `puppeteer` (for testing) ✅

## Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1 | 30 min | None |
| Phase 2 | 2 hours | Phase 1 complete |
| Phase 3 | 1-2 hours | Phase 2 complete |
| Phase 4 | 1 hour | Phase 3 complete |
| Phase 5 | 1 hour | Phase 4 complete |
| **Total** | **4-6 hours** | Sequential |

## How to Execute

### Quick Start
```bash
# 1. Read the phase files in order
cat PHASE_1_DEPENDENCIES.md
cat PHASE_2_MANIFEST_GENERATION.md
cat PHASE_3_FILTER_UI_FIX.md
cat PHASE_4_TESTING.md
cat PHASE_5_TAURI_INTEGRATION.md

# 2. Execute each phase
# Follow the detailed instructions in each phase file

# 3. Verify at each step
# Don't move to next phase until current phase is complete
```

### Progress Tracking
Mark each phase as complete by updating the status in each file:
- 🟡 In Progress
- ✅ Complete
- 🔴 Blocked

## Files to Create

1. `PHASE_1_DEPENDENCIES.md` - Dependency setup
2. `PHASE_2_MANIFEST_GENERATION.md` - Rewrite manifest script
3. `PHASE_3_FILTER_UI_FIX.md` - Fix filter clicks
4. `PHASE_4_TESTING.md` - End-to-end testing
5. `PHASE_5_TAURI_INTEGRATION.md` - Tauri integration

## Reference Documentation

- [marketplace-sdk README](../bin/79_marketplace_core/marketplace-sdk/README.md)
- [marketplace-node README](../bin/79_marketplace_core/marketplace-node/README.md)
- [HuggingFace API Docs](https://huggingface.co/docs/hub/api)
- [Next.js App Router Docs](https://nextjs.org/docs/app)

## Notes

- **WASM is BUILD TIME ONLY** - The frontend loads static JSON manifests
- **Tauri uses NATIVE Rust** - No WASM needed, direct SDK access
- **No premature celebration** - Test each phase thoroughly before moving on
- **Use existing types** - Don't create parallel type systems
- **Delete hacky code** - Remove my bad TypeScript fetch implementation
