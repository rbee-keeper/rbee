# Proper TypeScript Configurations

**Date:** 2025-11-12  
**Status:** ✅ COMPLETE - All lessons from turbo build fixes applied

## Overview

This package provides battle-tested TypeScript configurations for all rbee projects. These configs incorporate lessons learned from fixing 100+ TypeScript errors with proper solutions (no shortcuts).

## Available Configurations

### 1. `base.json` - Foundation for All Configs

**Use for:** Base configuration extended by all other configs

**Key Features:**
- ✅ `strict: true` - Full strict mode enabled
- ✅ `exactOptionalPropertyTypes: true` - Prevents `undefined` assignment to optional props
- ✅ `noUncheckedIndexedAccess: true` - Array access returns `T | undefined`
- ✅ `noImplicitOverride: true` - Explicit override keyword required
- ✅ `noUncheckedSideEffectImports: true` - Catch side-effect import issues

**Why these settings:**
- Catches real bugs at compile time
- Forces proper null/undefined handling
- Prevents common TypeScript pitfalls
- Aligns with modern TypeScript best practices

### 2. `library.json` - For TypeScript Libraries

**Use for:** Shared packages, utility libraries, non-React code

**Extends:** `base.json`

**Key Features:**
- Declaration files (`.d.ts`) generated
- Source maps for debugging
- ES2020 target for modern features
- Bundler module resolution

**Example usage:**
```json
{
  "extends": "@repo/typescript-config/library.json",
  "compilerOptions": {
    "rootDir": "./src",
    "outDir": "./dist"
  }
}
```

**Used by:**
- `@rbee/dev-utils`
- `@rbee/shared-config`
- `@rbee/marketplace-core`

### 3. `library-react.json` - For React Libraries

**Use for:** React component libraries, React hooks packages

**Extends:** `library.json`

**Key Features:**
- DOM types included
- JSX support (`react-jsx`)
- Node types for build tools
- React-specific optimizations

**Example usage:**
```json
{
  "extends": "@repo/typescript-config/library-react.json",
  "compilerOptions": {
    "rootDir": "./src",
    "outDir": "./dist"
  }
}
```

**Used by:**
- `@rbee/rbee-ui`
- `@rbee/react-hooks`
- `@rbee/iframe-bridge`
- `@rbee/queen-rbee-react`
- `@rbee/rbee-hive-react`
- `@rbee/llm-worker-react`

### 4. `nextjs.json` - For Next.js Applications

**Use for:** Next.js apps (commercial, user-docs)

**Extends:** `base.json`

**Key Features:**
- Next.js plugin integration
- Path aliases (`@/*`)
- Incremental compilation
- No emit (Next.js handles compilation)

**Example usage:**
```json
{
  "extends": "@repo/typescript-config/nextjs.json",
  "compilerOptions": {
    "types": ["node"]
  }
}
```

**Used by:**
- `apps/commercial` (CF Pages)
- `apps/user-docs` (CF Pages)

### 5. `cloudflare-worker.json` - For Cloudflare Workers

**Use for:** Cloudflare Workers (marketplace, global-worker-catalog, admin)

**Extends:** `base.json`

**Key Features:**
- Cloudflare Workers types (`@cloudflare/workers-types`)
- No Node.js types (Workers runtime is different)
- ES2022 target (Workers support modern JS)
- No emit (Wrangler handles compilation)

**Example usage:**
```json
{
  "extends": "@repo/typescript-config/cloudflare-worker.json"
}
```

**Used by:**
- `apps/marketplace` (CF Worker)
- `bin/80-global-worker-catalog` (CF Worker)
- `bin/90-admin` (CF Worker)

**IMPORTANT:** Cloudflare Workers do NOT have Node.js APIs. Only use `@cloudflare/workers-types`.

### 6. `cloudflare-pages.json` - For Cloudflare Pages (Next.js)

**Use for:** Next.js apps deployed to Cloudflare Pages

**Extends:** `nextjs.json`

**Key Features:**
- Both Node.js types (for build) and Cloudflare types (for runtime)
- Next.js configuration
- CF Pages environment support

**Example usage:**
```json
{
  "extends": "@repo/typescript-config/cloudflare-pages.json"
}
```

**Used by:**
- `apps/commercial` (if using CF Pages features)
- `apps/user-docs` (if using CF Pages features)

### 7. `react-app.json` - For React Applications (Vite)

**Use for:** Standalone React apps built with Vite

**Extends:** `base.json`

**Key Features:**
- Vite-specific settings
- React JSX support
- DOM types

**Used by:**
- Keeper UI
- Hive UI
- Worker UIs

### 8. `vite.json` - For Vite Applications

**Use for:** Non-React Vite apps

**Extends:** `base.json`

**Key Features:**
- Vite optimizations
- Modern module resolution

## Lessons Learned from Turbo Build Fixes

### 1. Don't Exclude Test Files

❌ **WRONG:**
```json
{
  "exclude": ["**/*.test.ts", "**/*.test.tsx"]
}
```

✅ **RIGHT:**
```json
{
  "exclude": ["node_modules", "dist"]
}
```

**Why:** Test files should be type-checked too. Create `test-setup.d.ts` for proper global types instead of excluding tests.

### 2. Don't Disable Strict Checks

❌ **WRONG:**
```json
{
  "compilerOptions": {
    "exactOptionalPropertyTypes": false,
    "skipLibCheck": true,
    "types": []
  }
}
```

✅ **RIGHT:**
```json
{
  "compilerOptions": {
    "exactOptionalPropertyTypes": true,
    "skipLibCheck": false,
    "types": ["node"]
  }
}
```

**Why:** Strict checks catch real bugs. Fix the code, don't disable the checks.

### 3. Handle Optional Properties Properly

With `exactOptionalPropertyTypes: true`, you cannot assign `undefined` to optional properties.

❌ **WRONG:**
```typescript
interface Config {
  debug?: boolean
}

const config: Config = {
  debug: options?.debug  // ❌ Error if options?.debug is undefined
}
```

✅ **RIGHT:**
```typescript
// Option 1: Conditional spread
const config: Config = {
  ...(options?.debug !== undefined ? { debug: options.debug } : {})
}

// Option 2: Conditional assignment
const config: Config = {}
if (options?.debug !== undefined) {
  config.debug = options.debug
}

// Option 3: Delete operator
config.debug = value
delete config.debug  // Remove property entirely
```

### 4. Handle Array Access Properly

With `noUncheckedIndexedAccess: true`, array access returns `T | undefined`.

❌ **WRONG:**
```typescript
const item = array[0]
doSomething(item)  // ❌ Error: item might be undefined
```

✅ **RIGHT:**
```typescript
// Option 1: Guard check
const item = array[0]
if (!item) throw new Error('Item not found')
doSomething(item)  // ✅ TypeScript knows it's defined

// Option 2: Nullish coalescing
const item = array[0] ?? defaultValue

// Option 3: Optional chaining
const value = array[0]?.property
```

### 5. Test Files Need Proper Setup

❌ **WRONG:**
```typescript
// Requires @types/node in production
global.window = { ... }
```

✅ **RIGHT:**
```typescript
// Create src/test-setup.d.ts
declare global {
  var global: typeof globalThis
}
export {}
```

## Migration Guide

### From Old Config to New Config

1. **Identify your project type:**
   - Library? → `library.json` or `library-react.json`
   - Next.js app? → `nextjs.json` or `cloudflare-pages.json`
   - CF Worker? → `cloudflare-worker.json`
   - React app? → `react-app.json`

2. **Update tsconfig.json:**
   ```json
   {
     "extends": "@repo/typescript-config/[config-name].json",
     "compilerOptions": {
       "rootDir": "./src",
       "outDir": "./dist"
     }
   }
   ```

3. **Remove shortcuts:**
   - Remove `types: []`
   - Remove `exactOptionalPropertyTypes: false`
   - Remove `skipLibCheck: true`
   - Remove test file exclusions

4. **Fix errors properly:**
   - Use patterns from "Lessons Learned" section
   - Create `test-setup.d.ts` for test files
   - Add proper null checks
   - Use conditional spreads for optional props

## Common Patterns

### Pattern 1: Mock Calls in Tests

```typescript
// ❌ WRONG
const [arg1, arg2] = mockFn.mock.calls[0]  // Error: might be undefined

// ✅ RIGHT
const firstCall = mockFn.mock.calls[0]
expect(firstCall).toBeDefined()
const [arg1, arg2] = firstCall ?? []
expect(arg1?.property).toBe(value)
```

### Pattern 2: Optional Props in Components

```typescript
// ❌ WRONG
<Component
  optionalProp={value}  // Error if value is undefined
/>

// ✅ RIGHT
<Component
  {...(value ? { optionalProp: value } : {})}
/>
```

### Pattern 3: Array Destructuring

```typescript
// ❌ WRONG
const parts = str.split('.')
const ext = parts[parts.length - 1].toUpperCase()  // Error: might be undefined

// ✅ RIGHT
const parts = str.split('.')
const ext = parts[parts.length - 1]
return ext ? ext.toUpperCase() : 'DEFAULT'
```

## Benefits

✅ **Type Safety** - Catches bugs at compile time  
✅ **Consistency** - Same patterns across all projects  
✅ **Maintainability** - Clear, documented patterns  
✅ **Future-Proof** - Works with stricter TypeScript versions  
✅ **No Shortcuts** - Proper fixes, not hacks  

## Support

For questions or issues with TypeScript configs, see:
- `.windsurf/TURBO_BUILD_PROPER_FIXES.md` - Detailed fix patterns
- This document - Configuration reference
- TypeScript docs - https://www.typescriptlang.org/tsconfig
