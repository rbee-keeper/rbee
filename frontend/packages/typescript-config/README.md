# @repo/typescript-config

**Battle-tested TypeScript configurations for rbee projects**

## Quick Start

Choose the config that matches your project type:

| Project Type | Config | Example |
|--------------|--------|---------|
| TypeScript Library | `library.json` | `@rbee/marketplace-core` |
| React Library | `library-react.json` | `@rbee/rbee-ui` |
| React App (Vite) | `react-app.json` | Keeper UI |
| Next.js App | `nextjs.json` | - |
| Cloudflare Worker | `cloudflare-worker.json` | `apps/marketplace` |
| Cloudflare Pages | `cloudflare-pages.json` | `apps/commercial` |

## Available Configs

### `base.json` - Foundation

**Don't use directly** - Extended by all other configs

**Features:**
- ✅ `strict: true` - Full strict mode
- ✅ `noUncheckedIndexedAccess: true` - Array access returns `T | undefined`
- ✅ `exactOptionalPropertyTypes: true` - No `undefined` assignment to optional props
- ✅ `noImplicitOverride: true` - Explicit override keyword required
- ✅ `noUncheckedSideEffectImports: true` - Catch side-effect import issues

### `library.json` - TypeScript Libraries

**Use for:** Shared packages, utility libraries, non-React code

```json
{
  "extends": "@repo/typescript-config/library.json",
  "compilerOptions": {
    "rootDir": "./src",
    "outDir": "./dist"
  }
}
```

**Examples:** `@rbee/dev-utils`, `@rbee/shared-config`

### `library-react.json` - React Libraries

**Use for:** React component libraries, React hooks packages

```json
{
  "extends": "@repo/typescript-config/library-react.json",
  "compilerOptions": {
    "rootDir": "./src",
    "outDir": "./dist"
  }
}
```

**Examples:** `@rbee/rbee-ui`, `@rbee/react-hooks`, `@rbee/iframe-bridge`

### `nextjs.json` - Next.js Applications

**Use for:** Next.js apps

```json
{
  "extends": "@repo/typescript-config/nextjs.json"
}
```

**Features:**
- Next.js plugin integration
- Path aliases (`@/*`)
- Incremental compilation
- No emit (Next.js handles compilation)

### `cloudflare-worker.json` - Cloudflare Workers

**Use for:** Cloudflare Workers (marketplace, global-worker-catalog, admin)

```json
{
  "extends": "@repo/typescript-config/cloudflare-worker.json"
}
```

**Features:**
- Cloudflare Workers types (`@cloudflare/workers-types`)
- No Node.js types (Workers runtime is different)
- ES2022 target
- React JSX support

**IMPORTANT:** Cloudflare Workers do NOT have Node.js APIs. Only use `@cloudflare/workers-types`.

### `cloudflare-pages.json` - Cloudflare Pages (Next.js)

**Use for:** Next.js apps deployed to Cloudflare Pages

```json
{
  "extends": "@repo/typescript-config/cloudflare-pages.json"
}
```

**Features:**
- Extends `nextjs.json`
- Both Node.js types (for build) and Cloudflare types (for runtime)
- CF Pages environment support

### `react-app.json` - React Applications (Vite)

**Use for:** Standalone React apps built with Vite

```json
{
  "extends": "@repo/typescript-config/react-app.json"
}
```

**Features:**
- Vite-specific settings
- React JSX support
- DOM types

### `vite.json` - Vite Config Files

**Use for:** `vite.config.ts` files

```json
{
  "extends": "@repo/typescript-config/vite.json"
}
```

## Key TypeScript Settings Explained

### `noUncheckedIndexedAccess: true`

Array access returns `T | undefined`:

```typescript
// ❌ WRONG
const item = array[0]
doSomething(item)  // Error: item might be undefined

// ✅ RIGHT
const item = array[0]
if (!item) throw new Error('Item not found')
doSomething(item)  // TypeScript knows it's defined
```

### `exactOptionalPropertyTypes: true`

Cannot assign `undefined` to optional properties:

```typescript
// ❌ WRONG
interface Config {
  debug?: boolean
}
const config: Config = {
  debug: options?.debug  // Error if options?.debug is undefined
}

// ✅ RIGHT - Conditional spread
const config: Config = {
  ...(options?.debug !== undefined ? { debug: options.debug } : {})
}
```

### `verbatimModuleSyntax: true`

Explicit type imports required:

```typescript
// ✅ Explicit type imports
import type { User } from './types'  // Type import
import { getUser } from './api'      // Value import
```

## Common Patterns

### Pattern 1: Array Access

```typescript
// ❌ WRONG
const parts = str.split('.')
const ext = parts[parts.length - 1].toUpperCase()  // Error: might be undefined

// ✅ RIGHT
const parts = str.split('.')
const ext = parts[parts.length - 1]
return ext ? ext.toUpperCase() : 'DEFAULT'
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

### Pattern 3: Test Files

Create `src/test-setup.d.ts` for test global types:

```typescript
/**
 * Test environment type declarations
 */
declare global {
  var global: typeof globalThis
}

export {}
```

## Migration from Old Configs

1. **Identify your project type** (see table above)
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
4. **Fix errors properly** using patterns above

## Benefits

✅ **Type Safety** - Catches bugs at compile time  
✅ **Consistency** - Same patterns across all projects  
✅ **Maintainability** - Clear, documented patterns  
✅ **Future-Proof** - Works with stricter TypeScript versions  
✅ **No Shortcuts** - Proper fixes, not hacks

## Resources

- [TypeScript TSConfig Reference](https://www.typescriptlang.org/tsconfig/)
- [Total TypeScript TSConfig Cheat Sheet](https://www.totaltypescript.com/tsconfig-cheat-sheet)
- [TypeScript 5.9 Release Notes](https://www.typescriptlang.org/docs/handbook/release-notes/typescript-5-9.html)

---

**Created by:** TEAM-471  
**Last Updated:** 2025-11-12
