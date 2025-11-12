# TypeScript Build Fixes Needed

**Date:** 2025-11-12  
**Status:** 🚧 IN PROGRESS

## Summary

After updating all tsconfig.json files to use the cleaned-up configs with strict settings (`exactOptionalPropertyTypes: true`, `noUncheckedIndexedAccess: true`), we have TypeScript errors to fix.

## Fixes Completed ✅

### 1. dev-utils & narration-client
**Issue:** Missing DOM types (uses `window`, `Location`)  
**Fix:** Changed from `library.json` to `library-react.json`  
**Files:**
- `packages/dev-utils/tsconfig.json`
- `packages/narration-client/tsconfig.json`

### 2. rbee-ui - GPUSelector.stories.tsx
**Issue:** `noUncheckedIndexedAccess` - array access returns `T | undefined`  
**Fix:** Added guard checks before `useState(array[0])`  
**Pattern:**
```typescript
// ❌ WRONG
const [selected, setSelected] = useState(sampleGPUs[0])

// ✅ RIGHT
const firstGPU = sampleGPUs[0]
if (!firstGPU) throw new Error('No GPUs available')
const [selected, setSelected] = useState(firstGPU)
```

### 3. rbee-ui - Navigation.tsx
**Issue:** `exactOptionalPropertyTypes` - cannot pass `undefined` to optional props  
**Fix:** Use conditional spreads  
**Pattern:**
```typescript
// ❌ WRONG
<Component optionalProp={value} />  // Error if value is undefined

// ✅ RIGHT
<Component {...(value ? { optionalProp: value } : {})} />
```

### 4. rbee-ui - TemplateBackground.test.ts
**Issue:** `noUncheckedIndexedAccess` - string index access  
**Fix:** Added guard check for `g[1]`  
**Pattern:**
```typescript
// ❌ WRONG
.replace(/-([a-z])/g, (g) => g[1].toUpperCase())

// ✅ RIGHT
.replace(/-([a-z])/g, (g) => {
  const char = g[1]
  return char ? char.toUpperCase() : ''
})
```

## Remaining Errors 🚧

### rbee-ui: 80 errors across 30 files

All are `exactOptionalPropertyTypes` errors where components receive `prop | undefined` but expect `prop` (optional).

**Affected files:**
- CheckItem.tsx (1 error)
- ContextMenu.tsx (1 error)
- DropdownMenu.tsx (1 error)
- Menubar.tsx (1 error)
- Slider.stories.tsx (4 errors)
- Slider.tsx (1 error)
- Sonner.tsx (1 error)
- Table.dark.stories.tsx (15 errors)
- Table.sticky.stories.tsx (12 errors)
- UseToast.ts (1 error)
- use-toast.ts (1 error)
- CategoryFilterBar.tsx (2 errors)
- CivitAIImageGallery.tsx (1 error)
- FilterBar.stories.tsx (14 errors)
- FilterBar/index.ts (1 error)
- ModelCardVertical.tsx (1 error)
- UniversalFilterBar.tsx (1 error)
- CivitAIModelDetail.tsx (1 error)
- HFModelDetail.tsx (4 errors)
- WorkerListTemplate.tsx (2 errors)
- MarkdownContent.tsx (1 error)
- NavLink.tsx (1 error)
- PlaybookAccordion.stories.tsx (1 error)
- PlaybookAccordion.tsx (1 error)
- ProvidersSecurityCard.tsx (1 error)
- SecurityGuarantees.tsx (1 error)
- StatusKPI.tsx (1 error)
- TemplateContainer.tsx (5 errors)
- UseCaseCard.tsx (1 error)
- EarningsCard.tsx (1 error)

## Fix Patterns

### Pattern 1: Conditional Spread for Optional Props
```typescript
// When passing optional props that might be undefined
<Component
  {...(optionalValue ? { optionalProp: optionalValue } : {})}
/>
```

### Pattern 2: Guard Check for Array Access
```typescript
// When accessing array elements
const item = array[0]
if (!item) throw new Error('Item not found')
// Now TypeScript knows item is defined
```

### Pattern 3: Nullish Coalescing for Defaults
```typescript
// When you have a default value
const value = array[0] ?? defaultValue
```

## Options

### Option 1: Fix All Errors (Recommended)
Systematically fix all 80 errors using the patterns above. This ensures maximum type safety.

**Pros:**
- Maximum type safety
- Catches real bugs at compile time
- Future-proof

**Cons:**
- Time-consuming (80 errors)
- Requires careful review of each case

### Option 2: Temporary Override for rbee-ui
Add to `packages/rbee-ui/tsconfig.json`:
```json
{
  "extends": "@repo/typescript-config/library-react.json",
  "compilerOptions": {
    "exactOptionalPropertyTypes": false,  // Temporary
    "rootDir": "./src",
    "outDir": "./dist"
  }
}
```

**Pros:**
- Unblocks build immediately
- Can fix errors incrementally

**Cons:**
- Loses type safety benefits
- Creates inconsistency with other packages

### Option 3: Create test-setup.d.ts
For test files only, create `src/test-setup.d.ts`:
```typescript
/**
 * Test environment type declarations
 */
declare global {
  var global: typeof globalThis
}

export {}
```

## Recommendation

**Fix all errors properly** using the patterns above. The strict settings catch real bugs and make the codebase more maintainable.

Start with the files that have the most errors:
1. Table.dark.stories.tsx (15 errors)
2. FilterBar.stories.tsx (14 errors)
3. Table.sticky.stories.tsx (12 errors)
4. TemplateContainer.tsx (5 errors)
5. HFModelDetail.tsx (4 errors)
6. Slider.stories.tsx (4 errors)

Then work through the rest systematically.

## Build Command

```bash
cd frontend
turbo build --filter=@rbee/ui
```

---

**Created by:** TEAM-472  
**Date:** 2025-11-12
