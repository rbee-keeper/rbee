# TypeScript Config Migration Guide

**Version:** 1.0.0 → 2.0.0  
**Date:** 2025-11-12

## What Changed

### New Configurations Added

1. **`cloudflare-worker.json`** - For CF Workers (marketplace, global-worker-catalog, admin)
2. **`cloudflare-pages.json`** - For Next.js apps on CF Pages (commercial, user-docs)

### Existing Configurations Updated

1. **`library.json`** - No longer excludes test files (we fix them properly now)
2. **`base.json`** - No changes (already perfect with strict settings)
3. **`library-react.json`** - Now includes `types: ["node"]` for proper Node.js support

## Migration Steps

### For Cloudflare Workers

**Before:**
```json
{
  "extends": "@repo/typescript-config/library.json",
  "compilerOptions": {
    "types": []  // ❌ Shortcut
  }
}
```

**After:**
```json
{
  "extends": "@repo/typescript-config/cloudflare-worker.json"
}
```

**Applies to:**
- `apps/marketplace`
- `bin/80-global-worker-catalog`
- `bin/90-admin`

### For Cloudflare Pages (Next.js)

**Before:**
```json
{
  "extends": "@repo/typescript-config/nextjs.json",
  "compilerOptions": {
    "types": ["node"]
  }
}
```

**After:**
```json
{
  "extends": "@repo/typescript-config/cloudflare-pages.json"
}
```

**Applies to:**
- `apps/commercial`
- `apps/user-docs`

### For React Libraries

**Before:**
```json
{
  "extends": "@repo/typescript-config/library-react.json",
  "compilerOptions": {
    "types": []  // ❌ Shortcut to avoid @types/node errors
  }
}
```

**After:**
```json
{
  "extends": "@repo/typescript-config/library-react.json",
  "compilerOptions": {
    "rootDir": "./src",
    "outDir": "./dist"
  }
}
```

**Note:** Now that `@types/node` is installed at workspace root, no need for `types: []` hack.

**Applies to:**
- `packages/rbee-ui`
- `packages/react-hooks`
- `packages/iframe-bridge`
- `bin/10_queen_rbee/ui/packages/queen-rbee-react`
- `bin/20_rbee_hive/ui/packages/rbee-hive-react`
- `bin/30_llm_worker_rbee/ui/packages/llm-worker-react`

### For TypeScript Libraries

**Before:**
```json
{
  "extends": "@repo/typescript-config/library.json",
  "compilerOptions": {
    "types": []  // ❌ Shortcut
  },
  "exclude": ["**/*.test.ts"]  // ❌ Excluding tests
}
```

**After:**
```json
{
  "extends": "@repo/typescript-config/library.json",
  "compilerOptions": {
    "rootDir": "./src",
    "outDir": "./dist"
  }
}
```

**Note:** Create `src/test-setup.d.ts` for test global types instead of excluding tests.

**Applies to:**
- `packages/dev-utils`
- `packages/shared-config`
- `packages/marketplace-core`
- `packages/sdk-loader`
- `packages/narration-client`

## Breaking Changes

### 1. Test Files No Longer Excluded

**Impact:** Test files will now be type-checked during build.

**Fix:** Create `src/test-setup.d.ts` in packages with tests:

```typescript
/**
 * Test environment type declarations
 */
declare global {
  var global: typeof globalThis
}

export {}
```

### 2. No More `types: []` Shortcut

**Impact:** Packages that used `types: []` to bypass type errors will now fail.

**Fix:** Install missing `@types` packages or use proper config:
- For libraries: Use `library.json` or `library-react.json`
- For CF Workers: Use `cloudflare-worker.json`
- For CF Pages: Use `cloudflare-pages.json`

### 3. Strict Type Checking Enforced

**Impact:** Code that relied on loose type checking will fail.

**Fix:** Apply proper patterns from `PROPER_TYPESCRIPT_CONFIGS.md`:
- Use conditional spreads for optional props
- Add null checks for array access
- Use proper type guards

## Verification

After migration, verify your build:

```bash
# Clean build
rm -rf dist .next out

# Build
pnpm build

# Should succeed with no TypeScript errors
```

## Rollback Plan

If you need to rollback:

```json
{
  "dependencies": {
    "@repo/typescript-config": "1.0.0"
  }
}
```

But please don't! The new configs are much better and catch real bugs.

## Support

- See `PROPER_TYPESCRIPT_CONFIGS.md` for detailed config documentation
- See `.windsurf/TURBO_BUILD_PROPER_FIXES.md` for fix patterns
- Ask in #engineering if you need help migrating
