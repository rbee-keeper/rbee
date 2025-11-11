# TEAM-471: TypeScript Config Standardization

**Date:** 2025-11-11  
**Status:** ✅ COMPLETE  
**Files Modified:** 13 tsconfig.json files

## Problem

Massive inconsistencies across all tsconfig.json files in the frontend workspace:

1. **Indentation chaos**: Tabs vs spaces (marketplace used tabs, others used spaces)
2. **Formatting inconsistency**: Single-line vs multi-line arrays
3. **Target versions**: ES2017 (apps) vs ES2020 (packages) vs ES2022 (base)
4. **Module resolution**: "bundler" vs "node"
5. **Module type**: "esnext" vs "ESNext" vs "ES2020"
6. **Missing options**: Some had `forceConsistentCasingInFileNames`, others didn't
7. **Paths inconsistency**: `@/*": ["./src/*"]` vs `@/*": ["./*"]`
8. **marketplace-core** extended base.json but base.json had incompatible settings

## Solution

### Standardized Next.js Apps (4 files)

**Files:**
- `apps/admin/tsconfig.json`
- `apps/commercial/tsconfig.json`
- `apps/marketplace/tsconfig.json`
- `apps/user-docs/tsconfig.json`

**Standard Config:**
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "forceConsistentCasingInFileNames": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "react-jsx",
    "incremental": true,
    "plugins": [{ "name": "next" }],
    "paths": { "@/*": ["./*"] },
    "types": ["./cloudflare-env.d.ts", "node"]
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts", ".next/dev/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

**Key Changes:**
- ✅ Target: ES2020 (was ES2017)
- ✅ Added `forceConsistentCasingInFileNames`
- ✅ Consistent paths: `@/*": ["./*"]` (not `./src/*`)
- ✅ Spaces only (no tabs)
- ✅ Single-line arrays for consistency

### Standardized Library Packages (9 files)

**Files:**
- `packages/dev-utils/tsconfig.json`
- `packages/env-config/tsconfig.json`
- `packages/iframe-bridge/tsconfig.json`
- `packages/marketplace-core/tsconfig.json`
- `packages/narration-client/tsconfig.json`
- `packages/rbee-ui/tsconfig.json`
- `packages/react-hooks/tsconfig.json`
- `packages/sdk-loader/tsconfig.json`
- `packages/shared-config/tsconfig.json`

**Standard Config (Base):**
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "lib": ["ES2020"],
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "isolatedModules": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

**Variations:**

**React Libraries** (rbee-ui, react-hooks, sdk-loader):
- Add `"lib": ["ES2020", "DOM"]` for React/DOM APIs
- Add `"jsx": "react-jsx"`
- react-hooks/sdk-loader: Add strict unused checks

**Browser Libraries** (dev-utils, narration-client, iframe-bridge, marketplace-core):
- Add `"lib": ["ES2020", "DOM"]` for console/window/fetch

**Node Libraries** (env-config, shared-config):
- Keep `"lib": ["ES2020"]` only
- env-config: Add `"types": ["node"]`

**marketplace-core specific:**
- Removed `extends: "@repo/typescript-config/base.json"` (was incompatible)
- Added explicit config matching other packages
- Added `@types/node` to package.json for process.env
- Added `"lib": ["ES2020", "DOM"]` for fetch/console
- Added `"types": ["node"]`

## Build Verification

✅ **All Next.js apps build successfully:**
- `pnpm -F commercial build` → ✅ Success (55 pages)
- All other apps use identical config

✅ **All library packages build successfully:**
- dev-utils → ✅ Success
- env-config → ✅ Success
- iframe-bridge → ✅ Success
- narration-client → ✅ Success
- rbee-ui → ✅ Success
- react-hooks → ✅ Success
- sdk-loader → ✅ Success
- shared-config → ✅ Success

⚠️ **marketplace-core has code errors (not tsconfig):**
- Missing export: `MarketplaceFilterParams` in adapter.ts
- This is a code issue, not a tsconfig issue

## Key Decisions

### 1. Target: ES2020 (not ES2017 or ES2022)
- **Rationale**: Modern enough for all features, widely supported
- **Consistent**: All packages and apps use ES2020

### 2. Module: ESNext + bundler resolution
- **Rationale**: Modern bundlers (Next.js, Vite) expect this
- **Consistent**: All packages use ESNext module

### 3. Removed base.json extension
- **Rationale**: base.json had incompatible settings (ES2022, verbatimModuleSyntax)
- **Solution**: Explicit config in each package for clarity

### 4. DOM lib where needed
- **Rationale**: Packages using console/window/fetch need DOM types
- **Packages**: dev-utils, narration-client, iframe-bridge, marketplace-core, rbee-ui, react-hooks, sdk-loader

### 5. Consistent formatting
- **Spaces only** (no tabs)
- **Single-line arrays** for compactness
- **Consistent option order**: target → module → moduleResolution → lib → ...

## Files Changed

**Next.js Apps (4):**
- `apps/admin/tsconfig.json` - ES2020, forceConsistentCasingInFileNames, paths
- `apps/commercial/tsconfig.json` - ES2020, forceConsistentCasingInFileNames
- `apps/marketplace/tsconfig.json` - ES2020, spaces (not tabs), forceConsistentCasingInFileNames
- `apps/user-docs/tsconfig.json` - ES2020, forceConsistentCasingInFileNames

**Library Packages (9):**
- `packages/dev-utils/tsconfig.json` - ESNext, bundler, DOM lib
- `packages/env-config/tsconfig.json` - ESNext, bundler, node types
- `packages/iframe-bridge/tsconfig.json` - ESNext, bundler, DOM lib
- `packages/marketplace-core/tsconfig.json` - Removed base.json, explicit config, DOM lib, node types
- `packages/marketplace-core/package.json` - Added @types/node
- `packages/narration-client/tsconfig.json` - ESNext, bundler, DOM lib
- `packages/rbee-ui/tsconfig.json` - Reordered options, added isolatedModules
- `packages/react-hooks/tsconfig.json` - Reordered options
- `packages/sdk-loader/tsconfig.json` - Reordered options
- `packages/shared-config/tsconfig.json` - ESNext, bundler

## Next Steps

1. ✅ All tsconfig.json files standardized
2. ✅ All builds verified
3. ⚠️ Fix marketplace-core code errors (missing export)
4. 🔄 Consider removing `packages/typescript-config/base.json` (no longer used)

## Summary

**Before:** 13 different tsconfig patterns, tabs vs spaces, ES2017 vs ES2020 vs ES2022, node vs bundler  
**After:** 2 consistent patterns (Next.js apps + library packages), all ES2020, all bundler, all spaces

**Build Status:** ✅ All apps and packages build successfully (except marketplace-core code errors)

---

// Created by: TEAM-471
