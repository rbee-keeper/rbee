# TEAM-467: Filter Centralization & CivitAI API Compliance

**Problem:** Filter constants and parsing logic were duplicated across multiple files, and NSFW levels were not using CivitAI API idiomatic values.

## Changes Made

### 1. Centralized ALL Filter Constants ✅

**File:** `/config/filter-constants.ts`

Now contains **BOTH** HuggingFace and CivitAI filter constants in ONE place:

```typescript
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// HUGGINGFACE FILTER CONSTANTS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

export const HF_SORTS = ['downloads', 'likes'] as const
export const HF_SIZES = ['all', 'small', 'medium', 'large'] as const
export const HF_LICENSES = ['all', 'apache', 'mit', 'other'] as const

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CIVITAI FILTER CONSTANTS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

export const CIVITAI_NSFW_LEVELS = ['all', 'pg', 'pg13', 'r', 'x'] as const
export const CIVITAI_TIME_PERIODS = ['all', 'week', 'month'] as const
export const CIVITAI_MODEL_TYPES = ['all', 'checkpoints', 'loras'] as const
export const CIVITAI_BASE_MODELS = ['all', 'sdxl', 'sd15'] as const
```

**Benefits:**
- ✅ Single source of truth for ALL filter values
- ✅ HuggingFace and CivitAI constants side-by-side
- ✅ Easy to compare and maintain
- ✅ Documented with API mappings

### 2. Created Shared Filter Parser ✅

**File:** `/config/filter-parser.ts`

Replaces duplicated parsing logic in `generate-model-manifests.ts`:

```typescript
// ❌ BEFORE - Duplicated in script
if (filterParts.includes('pg')) filters.nsfw.max_level = 'None'
if (filterParts.includes('pg13')) filters.nsfw.max_level = 'Soft'
if (filterParts.includes('r')) filters.nsfw.max_level = 'Mature'
if (filterParts.includes('x')) filters.nsfw.max_level = 'X'

// ✅ AFTER - Shared parser
const filters = parseCivitAIFilter(filter)
```

### 3. Verified CivitAI API Compliance ✅

**Source of Truth:** WASM contract from Rust SDK

```typescript
// From: /bin/79_marketplace_core/marketplace-node/wasm/marketplace_sdk.d.ts
export type NsfwLevel = "None" | "Soft" | "Mature" | "X" | "XXX"
```

**NSFW Level Mapping (CivitAI API Idiomatic):**

| URL Slug | API Enum | Numeric Values | Description |
|----------|----------|----------------|-------------|
| `pg` | `None` | `[1]` | PG only |
| `pg13` | `Soft` | `[1, 2]` | PG + PG-13 |
| `r` | `Mature` | `[1, 2, 4]` | Up to R-rated |
| `x` | `X` | `[1, 2, 4, 8]` | Up to X-rated |
| `all` | `XXX` | `[1, 2, 4, 8, 16]` | ALL levels (default) |

**Documentation Trail:**
1. **WASM Contract** (Rust → TypeScript): `/bin/79_marketplace_core/marketplace-node/wasm/marketplace_sdk.d.ts`
2. **SDK Implementation**: `/bin/79_marketplace_core/marketplace-node/src/civitai.ts`
3. **Filter Parser**: `/frontend/apps/marketplace/config/filter-parser.ts`
4. **Filter Constants**: `/frontend/apps/marketplace/config/filter-constants.ts`

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ filter-constants.ts                                             │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ SINGLE SOURCE OF TRUTH                                          │
│                                                                 │
│ HuggingFace:                  CivitAI:                         │
│ - HF_SORTS                    - CIVITAI_NSFW_LEVELS           │
│ - HF_SIZES                    - CIVITAI_TIME_PERIODS          │
│ - HF_LICENSES                 - CIVITAI_MODEL_TYPES           │
│                               - CIVITAI_BASE_MODELS            │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │
                    ┌─────────┴─────────┐
                    │                   │
┌───────────────────▼─────┐   ┌────────▼──────────────────┐
│ filter-parser.ts        │   │ filters.ts                │
│ ━━━━━━━━━━━━━━━━━━━━━━ │   │ ━━━━━━━━━━━━━━━━━━━━━━━━ │
│ Parsing Logic           │   │ Filter Generation         │
│                         │   │                           │
│ parseCivitAIFilter()    │   │ generateAllCivitAI...()  │
│ - Maps URL slugs        │   │ generateAllHF...()       │
│ - Returns API params    │   │                           │
└─────────────────────────┘   └───────────────────────────┘
            ▲                             ▲
            │                             │
            └─────────┬───────────────────┘
                      │
        ┌─────────────▼──────────────────┐
        │ generate-model-manifests.ts    │
        │ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
        │ Manifest Generation            │
        │                                │
        │ Uses shared parser + constants │
        └────────────────────────────────┘
```

## Files Changed

### Created
- ✅ `/config/filter-parser.ts` - Shared parsing logic
- ✅ `/config/filter-constants.ts` - Updated with both HF + CivitAI constants

### Modified
- ✅ `/scripts/generate-model-manifests.ts` - Uses shared parser
- ✅ `/bin/79_marketplace_core/marketplace-node/src/index.ts` - Removed error swallowing
- ✅ `/bin/79_marketplace_core/marketplace-node/src/civitai.ts` - Added NSFW validation

## Verification

**CivitAI API Documentation:**
- GitHub Wiki: https://github.com/civitai/civitai/wiki/REST-API-Reference
- NSFW levels are **numeric** in the API: `1, 2, 4, 8, 16, 32`
- Our SDK correctly maps enum values to these numbers

**WASM Contract:**
```bash
cat /home/vince/Projects/rbee/bin/79_marketplace_core/marketplace-node/wasm/marketplace_sdk.d.ts | grep -A 1 "NsfwLevel"
# export type NsfwLevel = "None" | "Soft" | "Mature" | "X" | "XXX";
```

## Benefits

### Before ❌
- Filter constants duplicated in multiple files
- Parsing logic hardcoded in script
- No clear source of truth
- HF and CivitAI constants in separate files
- Potential for drift between definitions

### After ✅
- **ONE file** for ALL filter constants (HF + CivitAI)
- **ONE parser** for filter paths → API params
- **Documented** with CivitAI API references
- **Type-safe** with WASM contract types
- **Easy to maintain** - change in one place

## Rule Zero Compliance

✅ **Breaking changes > backwards compatibility**
- Removed duplicated parsing logic
- Consolidated constants into single file
- No `_v2` or `_new` functions - just updated existing code

✅ **Single source of truth**
- All filter constants in `filter-constants.ts`
- All parsing logic in `filter-parser.ts`
- No duplication across codebase

✅ **Delete deprecated code**
- Removed inline parsing from script
- No legacy constants left behind
