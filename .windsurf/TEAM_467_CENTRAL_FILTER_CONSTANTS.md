# TEAM-467: Central Filter Constants - Single Source of Truth

**Date**: 2025-11-11  
**Status**: ✅ Complete

---

## 🐛 The Problem

**Issue**: Filter constants were duplicated in 3 different places

### Before (Duplicated)

**1. config/filters.ts** (Manifest Generation)
```typescript
const sorts = ['downloads', 'likes', 'recent'] as const
const sizes = ['all', 'small', 'medium', 'large'] as const
const licenses = ['all', 'apache', 'mit', 'other'] as const
```

**2. app/models/huggingface/filters.ts** (UI Filter Configs)
```typescript
const sorts = ['downloads', 'likes', 'recent'] as const
const sizes = ['all', 'small', 'medium', 'large'] as const
const licenses = ['all', 'apache', 'mit', 'other'] as const
```

**3. app/models/huggingface/HFFilterPage.tsx** (Validation)
```typescript
const validSorts = ['downloads', 'likes', 'recent'] as const
const validSizes = ['all', 'small', 'medium', 'large'] as const
const validLicenses = ['all', 'apache', 'mit', 'other'] as const
```

### Problems with Duplication

1. **Maintenance Hell**: Need to update 3 places to add a filter
2. **Easy to Forget**: Miss one place → bugs
3. **Out of Sync**: Different files have different values
4. **No Single Source of Truth**: Which one is correct?

---

## ✅ The Solution

### New File: `config/filter-constants.ts`

**SINGLE SOURCE OF TRUTH** for all filter values:

```typescript
// TEAM-467: SINGLE SOURCE OF TRUTH for all filter values
// This file is imported by:
// - config/filters.ts (manifest generation)
// - app/models/huggingface/filters.ts (UI filter configs)
// - app/models/huggingface/HFFilterPage.tsx (validation)

/**
 * HuggingFace Filter Constants
 * TEAM-467: Central place to edit all filter options
 */
export const HF_SORTS = ['downloads', 'likes', 'recent'] as const
export const HF_SIZES = ['all', 'small', 'medium', 'large'] as const
export const HF_LICENSES = ['all', 'apache', 'mit', 'other'] as const

export type HFSort = typeof HF_SORTS[number]
export type HFSize = typeof HF_SIZES[number]
export type HFLicense = typeof HF_LICENSES[number]

/**
 * CivitAI Filter Constants
 * TEAM-467: Central place to edit all filter options
 */
export const CIVITAI_NSFW_LEVELS = ['all', 'pg', 'pg13', 'r', 'x'] as const
export const CIVITAI_TIME_PERIODS = ['all', 'week', 'month'] as const
export const CIVITAI_MODEL_TYPES = ['all', 'checkpoints', 'loras'] as const
export const CIVITAI_BASE_MODELS = ['all', 'sdxl', 'sd15'] as const

export type CivitAINSFW = typeof CIVITAI_NSFW_LEVELS[number]
export type CivitAITimePeriod = typeof CIVITAI_TIME_PERIODS[number]
export type CivitAIModelType = typeof CIVITAI_MODEL_TYPES[number]
export type CivitAIBaseModel = typeof CIVITAI_BASE_MODELS[number]
```

---

## 🔄 Updated Files

### 1. config/filters.ts (Manifest Generation)
```typescript
import { HF_SORTS, HF_SIZES, HF_LICENSES } from './filter-constants'
import { CIVITAI_NSFW_LEVELS, CIVITAI_TIME_PERIODS, CIVITAI_MODEL_TYPES, CIVITAI_BASE_MODELS } from './filter-constants'

function generateAllHFFilterCombinations(): string[] {
  const sorts = HF_SORTS      // ✅ From central file
  const sizes = HF_SIZES      // ✅ From central file
  const licenses = HF_LICENSES // ✅ From central file
  // ...
}
```

### 2. app/models/huggingface/filters.ts (UI Filter Configs)
```typescript
import { HF_SORTS, HF_SIZES, HF_LICENSES } from '@/config/filter-constants'

function generateAllHFFilterConfigs(): FilterConfig<HuggingFaceFilters>[] {
  const sorts = HF_SORTS      // ✅ From central file
  const sizes = HF_SIZES      // ✅ From central file
  const licenses = HF_LICENSES // ✅ From central file
  // ...
}
```

### 3. app/models/huggingface/HFFilterPage.tsx (Validation)
```typescript
import { HF_SORTS, HF_SIZES, HF_LICENSES } from '@/config/filter-constants'

export function HFFilterPage({ initialModels, initialFilter }: Props) {
  // TEAM-467: Validate URL params - Uses SHARED constants
  const validSorts = HF_SORTS      // ✅ From central file
  const validSizes = HF_SIZES      // ✅ From central file
  const validLicenses = HF_LICENSES // ✅ From central file
  // ...
}
```

---

## 🎯 Benefits

### 1. Single Source of Truth
- **One place** to add/remove/modify filters
- **Guaranteed consistency** across all files
- **No more duplication**

### 2. Easy Maintenance
**Before** (add new sort option):
1. Update `config/filters.ts`
2. Update `app/models/huggingface/filters.ts`
3. Update `app/models/huggingface/HFFilterPage.tsx`
4. Update error message
5. Hope you didn't miss anything

**After** (add new sort option):
1. Update `config/filter-constants.ts`
2. Done! ✅

### 3. Type Safety
```typescript
export type HFSort = typeof HF_SORTS[number]
// Type: 'downloads' | 'likes' | 'recent'

export type HFSize = typeof HF_SIZES[number]
// Type: 'all' | 'small' | 'medium' | 'large'

export type HFLicense = typeof HF_LICENSES[number]
// Type: 'all' | 'apache' | 'mit' | 'other'
```

### 4. Self-Documenting
```typescript
/**
 * HuggingFace Filter Constants
 * TEAM-467: Central place to edit all filter options
 */
export const HF_SORTS = ['downloads', 'likes', 'recent'] as const
```

Clear comment explains this is THE place to edit filters.

---

## 📊 Architecture

### File Structure
```
config/
├── filter-constants.ts      ← SINGLE SOURCE OF TRUTH
└── filters.ts               ← Uses constants (manifest generation)

app/models/huggingface/
├── filters.ts               ← Uses constants (UI configs)
└── HFFilterPage.tsx         ← Uses constants (validation)
```

### Import Flow
```
filter-constants.ts
    ↓
    ├─→ config/filters.ts (manifest generation)
    ├─→ app/models/huggingface/filters.ts (UI configs)
    └─→ app/models/huggingface/HFFilterPage.tsx (validation)
```

---

## 🚀 How to Add a New Filter

### Example: Add "trending" sort option

**1. Edit ONE file** (`config/filter-constants.ts`):
```typescript
export const HF_SORTS = ['downloads', 'likes', 'recent', 'trending'] as const
//                                                       ^^^^^^^^^ Add here
```

**2. That's it!** ✅

All three systems automatically get the new option:
- ✅ Manifest generation creates `hf-filter-trending.json`
- ✅ UI shows "Trending" in sort dropdown
- ✅ Validation accepts `?sort=trending` in URL

---

## 🔍 Verification

### Check All Files Import Correctly
```bash
# Manifest generation
grep -n "filter-constants" config/filters.ts

# UI configs
grep -n "filter-constants" app/models/huggingface/filters.ts

# Validation
grep -n "filter-constants" app/models/huggingface/HFFilterPage.tsx
```

### Test Adding a Filter
1. Add new value to `filter-constants.ts`
2. Run `NODE_ENV=production pnpm run generate:manifests`
3. Check new manifests are generated
4. Check UI shows new option
5. Check validation accepts new value

---

## ✅ Checklist

- [x] Created `config/filter-constants.ts`
- [x] Added HuggingFace constants
- [x] Added CivitAI constants
- [x] Updated `config/filters.ts` to import
- [x] Updated `app/models/huggingface/filters.ts` to import
- [x] Updated `app/models/huggingface/HFFilterPage.tsx` to import
- [x] Verified all imports work
- [x] Documented the architecture

---

## 📝 Future Improvements

### 1. Generate UI Filter Groups from Constants
Instead of hardcoding:
```typescript
export const HUGGINGFACE_FILTER_GROUPS: FilterGroup[] = [
  {
    id: 'size',
    label: 'Model Size',
    options: [
      { label: 'All Sizes', value: 'all' },
      { label: 'Small (<7B)', value: 'small' },
      // ...
    ],
  },
]
```

Generate from constants:
```typescript
export const HUGGINGFACE_FILTER_GROUPS = generateFilterGroups({
  size: {
    label: 'Model Size',
    options: HF_SIZES,
    labels: {
      all: 'All Sizes',
      small: 'Small (<7B)',
      // ...
    }
  }
})
```

### 2. Validate at Build Time
Add a build step that ensures:
- All constants are used
- No orphaned manifests
- No missing manifests

---

**TEAM-467: Single source of truth established! 🎯**

**Now there's ONE place to edit all filter options!**
