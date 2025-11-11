# TEAM-477: exactOptionalPropertyTypes Fix

**Date:** 2025-11-11  
**Status:** ✅ FIXED (HuggingFace & CivitAI)  
**Remaining:** Other rbee-ui type issues (54 errors in 38 files)

## Problem

TypeScript error with `exactOptionalPropertyTypes: true`:

```
Type '{ limit: number; library?: HuggingFaceListModelsParams["library"]; ... }' 
is not assignable to type 'HuggingFaceListModelsParams' with 'exactOptionalPropertyTypes: true'.
Types of property 'sort' are incompatible.
Type 'HuggingFaceSort | undefined' is not assignable to type 'HuggingFaceSort'.
```

### Root Cause

When `exactOptionalPropertyTypes: true` is enabled in tsconfig, optional properties must **explicitly include `undefined`** in their type union if you might pass `undefined` values.

**The issue:** Conditional spreading like `...(condition && { prop: value })` can introduce `undefined`, but the interface didn't allow it.

## Solution Applied (RULE ZERO)

✅ **Fixed the type definitions** (not the consuming code)  
✅ **Updated both HuggingFace and CivitAI interfaces**  
✅ **Removed unnecessary vite/client types from library package**

### Files Modified

**1. `/packages/marketplace-core/src/adapters/huggingface/types.ts`**
- Added `| undefined` to all optional properties in `HuggingFaceListModelsParams`
- Example: `sort?: HuggingFaceSort` → `sort?: HuggingFaceSort | undefined`

**2. `/packages/marketplace-core/src/adapters/civitai/types.ts`**
- Added `| undefined` to all optional properties in `CivitAIListModelsParams`
- Prevents same issue in CivitAI pages

**3. `/packages/marketplace-core/tsconfig.json`**
- Removed `"vite/client"` from types array (not needed for library packages)
- Fixed build error: `Cannot find type definition file for 'vite/client'`

## Verification

✅ `@rbee/marketplace-core` builds successfully  
✅ HuggingFace page error on line 14 is FIXED  
✅ CivitAI pages won't have same issue  

## Why This Follows RULE ZERO

**RULE ZERO:** Breaking changes > backwards compatibility

Instead of:
- ❌ Creating workarounds in consuming code
- ❌ Adding type assertions everywhere
- ❌ Disabling `exactOptionalPropertyTypes`

We:
- ✅ **Fixed the root type definition**
- ✅ **Made it work correctly with strict TypeScript**
- ✅ **Prevented future issues**

The type change is **not breaking** - it's more accurate. Optional properties already allowed `undefined` at runtime, we just made the types match reality.

## Remaining Work

There are 54 other `exactOptionalPropertyTypes` errors in rbee-ui components. These need similar fixes:

- Navigation component icon props
- IconCardHeader subtitle props
- Various UI component optional props

**Recommendation:** Fix these systematically by updating the type definitions in rbee-ui, not by adding workarounds.

## Technical Details

### Before (WRONG)
```typescript
export interface HuggingFaceListModelsParams {
  sort?: HuggingFaceSort  // Doesn't explicitly allow undefined
}
```

### After (CORRECT)
```typescript
export interface HuggingFaceListModelsParams {
  sort?: HuggingFaceSort | undefined  // Explicitly allows undefined
}
```

### Why This Matters

With `exactOptionalPropertyTypes: true`, TypeScript distinguishes between:
- `{ prop?: T }` - Property can be omitted OR have type T
- `{ prop?: T | undefined }` - Property can be omitted, have type T, OR be explicitly undefined

When you use conditional spreading, you might set the property to `undefined`, so you need the second form.

## Build Commands

```bash
# Rebuild marketplace-core
pnpm --filter @rbee/marketplace-core build

# Verify marketplace app (will show other errors, but not line 14)
pnpm --filter marketplace typecheck
```

---

**TEAM-477 signature:** Fixed exactOptionalPropertyTypes compatibility in marketplace-core
