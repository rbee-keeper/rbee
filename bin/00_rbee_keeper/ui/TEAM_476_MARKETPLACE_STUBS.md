# TEAM-476: Marketplace Stubs for Keeper UI

**Date:** 2025-11-11  
**Status:** ✅ COMPLETE  
**Reason:** Deleted marketplace-sdk and marketplace-node to restart with client-side fetcher

## What Changed

### 1. Removed Dependencies

**Removed from `/bin/00_rbee_keeper/ui/package.json`:**
```json
"@rbee/marketplace-node": "workspace:*"
```

**Removed from `/frontend/packages/rbee-ui/package.json`:**
```json
"@rbee/marketplace-node": "workspace:*"
```

### 2. Created Stub File

**Created:** `/bin/00_rbee_keeper/ui/src/lib/marketplace-stubs.ts`

This file provides temporary stubs for all marketplace-node imports:

#### Types
- `TimePeriod` - CivitAI time periods
- `CivitaiModelType` - CivitAI model types
- `BaseModel` - CivitAI base models
- `CivitaiSort` - CivitAI sort options
- `HuggingFaceSort` - HuggingFace sort options
- `FilterableModel` - Generic model interface

#### Constants
- `CIVITAI_DEFAULTS` - Default CivitAI filter values
- `HF_DEFAULTS` - Default HuggingFace filter values
- `HF_SIZES` - HuggingFace model sizes
- `HF_LICENSES` - HuggingFace licenses

#### Functions (Stubbed)
- `applyCivitAIFilters()` - TODO: Implement client-side filtering
- `applyHuggingFaceFilters()` - TODO: Implement client-side filtering
- `buildHuggingFaceFilterDescription()` - TODO: Implement description builder

### 3. Updated Imports

**Updated Files:**
- `/bin/00_rbee_keeper/ui/src/pages/MarketplaceCivitai.tsx`
- `/bin/00_rbee_keeper/ui/src/pages/MarketplaceHuggingFace.tsx`

**Changed:**
```typescript
// OLD
import { ... } from '@rbee/marketplace-node'

// NEW
import { ... } from '@/lib/marketplace-stubs'
```

## Current State

✅ **Build passes** - `pnpm i` succeeds  
✅ **No broken imports** - All imports point to stubs  
⚠️ **Filtering not implemented** - Stub functions just return all models  
⚠️ **Console warnings** - Stubs log warnings when called  

## Next Steps

### TODO: Implement Client-Side Fetcher

Replace the stubs with actual client-side implementation:

1. **Create client-side fetchers:**
   - `/bin/00_rbee_keeper/ui/src/lib/civitai-client.ts`
   - `/bin/00_rbee_keeper/ui/src/lib/huggingface-client.ts`

2. **Implement filtering logic:**
   - `applyCivitAIFilters()` - Filter by time, type, base model, sort
   - `applyHuggingFaceFilters()` - Filter by size, license, sort
   - `buildHuggingFaceFilterDescription()` - Build filter description

3. **Add API calls:**
   - Direct fetch to CivitAI API: `https://civitai.com/api/v1/models`
   - Direct fetch to HuggingFace API: `https://huggingface.co/api/models`

4. **Update Tauri commands:**
   - Remove server-side marketplace commands
   - Use client-side fetching instead

## Architecture Change

### Before (Server-Side)
```
Keeper UI
  ↓
Tauri Command (marketplace_list_civitai_models)
  ↓
marketplace-node (Rust WASM)
  ↓
marketplace-sdk (Rust)
  ↓
CivitAI API
```

### After (Client-Side)
```
Keeper UI
  ↓
Client-Side Fetcher (TypeScript)
  ↓
CivitAI API (direct fetch)
```

## Benefits of Client-Side Approach

✅ **Simpler** - No Rust WASM compilation  
✅ **Faster iteration** - TypeScript changes don't require rebuild  
✅ **Smaller bundle** - No WASM overhead  
✅ **Easier debugging** - Browser DevTools work natively  
✅ **More flexible** - Can use browser APIs directly  

## Files Modified

1. `/bin/00_rbee_keeper/ui/package.json` - Removed dependency
2. `/frontend/packages/rbee-ui/package.json` - Removed dependency
3. `/bin/00_rbee_keeper/ui/src/lib/marketplace-stubs.ts` - Created stubs
4. `/bin/00_rbee_keeper/ui/src/pages/MarketplaceCivitai.tsx` - Updated imports
5. `/bin/00_rbee_keeper/ui/src/pages/MarketplaceHuggingFace.tsx` - Updated imports

## Verification

```bash
# Install dependencies
pnpm i

# Build should pass (with stub warnings)
pnpm --filter @rbee/keeper-ui build
```

---

**TEAM-476 RULE ZERO:** We're restarting the marketplace implementation with a client-side fetcher instead of the complex Rust WASM approach. The stubs are temporary placeholders until the client-side implementation is complete.
