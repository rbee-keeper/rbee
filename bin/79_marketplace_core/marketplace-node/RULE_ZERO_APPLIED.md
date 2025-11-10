# Rule Zero Applied: ModelFile Type Deduplication

**TEAM-463: Eliminated Duplicate ModelFile Definitions**  
**Date:** 2025-11-10  
**Status:** ✅ COMPLETE

## Problem: Type Inconsistency

**Root Cause:** HuggingFace API uses `rfilename` (relative filename) in their JSON responses.

```json
{
  "modelId": "openai-community/gpt2",
  "siblings": [
    { "rfilename": "config.json", "size": 665 }
  ]
}
```

This led to inconsistency:
- ❌ Some code used `rfilename` (HuggingFace API format)
- ❌ Some code used `filename` (our canonical format)
- ❌ Multiple duplicate type definitions

## Rule Zero Solution

**RULE ZERO: Delete duplicate code, normalize at the boundary.**

### Architecture

```
External API (HuggingFace)
  ↓ uses rfilename
RAW API Types (HFModel)
  ↓ normalize at boundary
artifacts-contract::ModelFile (filename) ← SOURCE OF TRUTH
  ↓ re-exported by
marketplace-sdk
  ↓ generates WASM types
marketplace-node (imports from WASM)
```

### Changes Made

**1. Deleted Duplicate Type Definition**

❌ **BEFORE** (`src/types.ts`):
```typescript
export interface ModelFile {
  rfilename: string  // WRONG - duplicate definition
  size: number
}
```

✅ **AFTER** (`src/types.ts`):
```typescript
// Import from WASM SDK (source: artifacts-contract)
import type { ModelFile as WasmModelFile } from '../wasm/marketplace_sdk'
export type ModelFile = WasmModelFile
```

**2. Normalized at Boundary**

✅ **Conversion** (`src/index.ts`):
```typescript
function convertHFModel(hf: HFModel): Model {
  return {
    // ...
    // TEAM-463: Convert HuggingFace's rfilename → our canonical filename
    siblings: hf.siblings?.map((s) => ({ 
      filename: s.rfilename,  // ← Normalize here
      size: s.size || 0 
    })) || [],
  }
}
```

**3. Documented Raw API Format**

✅ **HFModel** (`src/huggingface.ts`):
```typescript
/**
 * RAW HuggingFace API response
 * 
 * ⚠️ This represents the EXTERNAL API format from HuggingFace.
 * HuggingFace uses `rfilename` (relative filename) in their API.
 * 
 * We normalize this to our canonical `filename` format when converting to our Model type.
 */
export interface HFModel {
  siblings?: Array<{
    rfilename: string  // ← OK - this is the raw API format
    size?: number
  }>
}
```

## Type Flow

```
HuggingFace API
  ↓ returns rfilename
HFModel (raw API type)
  ↓ convertHFModel() normalizes
Model (our type with filename)
  ↓ uses
ModelFile from artifacts-contract
```

## Benefits

✅ **Single Source of Truth** - `ModelFile` defined once in `artifacts-contract`  
✅ **No Duplicates** - Deleted duplicate type definition  
✅ **Normalized** - External API format converted at boundary  
✅ **Type Safe** - TypeScript enforces correct usage  
✅ **Documented** - Clear comments explain raw vs normalized formats  

## Verification

```bash
# TypeScript compilation
cd bin/79_marketplace_core/marketplace-node
npx tsc --noEmit  # ✅ Passes

# Contract compilation
cargo check -p artifacts-contract  # ✅ Passes

# Marketplace SDK
cargo check -p marketplace-sdk  # ✅ Passes
```

## Key Principle

**Normalize external formats at the boundary, use canonical format internally.**

- ✅ **External APIs** can use whatever field names they want (`rfilename`)
- ✅ **Raw API types** (like `HFModel`) represent the external format
- ✅ **Conversion functions** normalize to our canonical format (`filename`)
- ✅ **Internal types** use the canonical format everywhere

This prevents type inconsistency from spreading through the codebase.

## See Also

- `MODELFILE_CONTRACT.md` - ModelFile type contract documentation
- `bin/97_contracts/artifacts-contract/` - Source of truth for types
