# @repo/typescript-config v2.0.0

**Battle-tested TypeScript configurations for rbee projects**

Incorporates lessons learned from fixing 100+ TypeScript errors with proper solutions (no shortcuts).

## 🚀 Quick Start

Choose the config that matches your project type:

| Project Type | Config | Example |
|--------------|--------|---------|
| TypeScript Library | `library.json` | `@rbee/marketplace-core` |
| React Library | `library-react.json` | `@rbee/rbee-ui` |
| React App (Vite) | `react-app.json` | Keeper UI |
| Next.js App | `nextjs.json` | - |
| **Cloudflare Worker** | `cloudflare-worker.json` | `apps/marketplace` |
| **Cloudflare Pages** | `cloudflare-pages.json` | `apps/commercial` |

## 📦 Available Configs

### `base.json` - Universal Base Config
**Use for:** Foundation for all configs (don't use directly)

**Features:**
- ✅ **Maximum Strictness**: `strict`, `noUncheckedIndexedAccess`, `exactOptionalPropertyTypes`
- ✅ **Type Safety**: `verbatimModuleSyntax`, `isolatedModules`, `noImplicitOverride`
- ✅ **Modern TS**: `noUncheckedSideEffectImports` for safer imports
- ✅ **ES2022**: Modern JavaScript features

---

### `cloudflare-worker.json` - Cloudflare Workers ⭐ NEW
**Use for:** CF Workers (marketplace, global-worker-catalog, admin)

```json
{
  "extends": "@repo/typescript-config/cloudflare-worker.json"
}
```

**Features:**
- ✅ Cloudflare Workers types (`@cloudflare/workers-types`)
- ✅ No Node.js types (Workers runtime is different)
- ✅ React JSX support for components
- ✅ ES2022 target

**Used by:** `apps/marketplace`, `bin/80-global-worker-catalog`, `bin/90-admin`

---

### `cloudflare-pages.json` - Cloudflare Pages (Next.js) ⭐ NEW
**Use for:** Next.js apps deployed to CF Pages

```json
{
  "extends": "@repo/typescript-config/cloudflare-pages.json"
}
```

**Features:**
- ✅ Extends `nextjs.json`
- ✅ Both Node.js types (build) and CF types (runtime)
- ✅ CF Pages environment support

**Used by:** `apps/commercial`, `apps/user-docs`

---

### `nextjs.json` - Next.js Applications
**Use for:** Next.js apps

```json
{
  "extends": "@repo/typescript-config/nextjs.json"
}
```

**Includes:**
- DOM types for browser APIs
- Next.js plugin support
- Path aliases (`@/*`)
- Incremental compilation

---

### `react-app.json` - React Applications (Vite)
**Use for:** React apps built with Vite

```json
{
  "extends": "@repo/typescript-config/react-app.json"
}
```

**Includes:**
- DOM types
- React JSX transform
- Vite client types
- `useDefineForClassFields` for modern class fields

---

### `library.json` - TypeScript Libraries
**Use for:** Pure TypeScript libraries (no React, no DOM)

```json
{
  "extends": "@repo/typescript-config/library.json"
}
```

**Includes:**
- Declaration file generation (`.d.ts`)
- Source maps
- Output to `dist/`
- ES2020 target for broad compatibility

**Examples:** `env-config`, `shared-config`, `sdk-loader`

---

### `library-react.json` - React Libraries
**Use for:** React component libraries

```json
{
  "extends": "@repo/typescript-config/library-react.json"
}
```

**Includes:**
- Everything from `library.json`
- DOM types
- React JSX transform
- Node types

**Examples:** `rbee-ui`, `react-hooks`, `marketplace-core`

---

### `vite.json` - Vite Config Files
**Use for:** `vite.config.ts` files

```json
{
  "extends": "@repo/typescript-config/vite.json"
}
```

**Includes:**
- Composite project support
- Node types
- Build info caching

---

## 🎯 Migration Guide

### From Old Configs

**Before (inconsistent):**
```json
{
  "compilerOptions": {
    "target": "ES2017",
    "module": "ES2020",
    "moduleResolution": "node"
  }
}
```

**After (modern):**
```json
{
  "extends": "@repo/typescript-config/nextjs.json"
}
```

### Next.js Apps

Replace your `tsconfig.json` with:

```json
{
  "extends": "@repo/typescript-config/nextjs.json",
  "compilerOptions": {
    "types": ["./cloudflare-env.d.ts", "node"]
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

### TypeScript Libraries

Replace your `tsconfig.json` with:

```json
{
  "extends": "@repo/typescript-config/library.json"
}
```

### React Libraries

Replace your `tsconfig.json` with:

```json
{
  "extends": "@repo/typescript-config/library-react.json",
  "compilerOptions": {
    "types": ["node", "vite/client"]
  }
}
```

---

## 🔥 Key Features

### 1. Maximum Type Safety

```typescript
// ✅ Catches array access errors
const arr = [1, 2, 3];
const item = arr[10]; // Type: number | undefined (not just number!)

// ✅ Requires override keyword
class Base {
  method() {}
}
class Child extends Base {
  override method() {} // Required!
}

// ✅ Exact optional properties
type Config = {
  port?: number;
};
const config: Config = { port: undefined }; // Error! Must omit or provide number
```

### 2. Modern Module System

```typescript
// ✅ Forces explicit type imports
import type { User } from './types'; // Required for type-only imports
import { getUser } from './api';     // Value import

// ✅ Works with all bundlers
// module: preserve + moduleResolution: bundler
```

### 3. Side Effect Import Safety

```typescript
// ✅ Catches unintended side effects (TS 5.9+)
import './polyfill'; // Warning if polyfill doesn't export anything
```

---

## 📊 Comparison with Old Config

| Feature | Old `base.json` | New `base.json` |
|---------|----------------|-----------------|
| Target | ES2022 | es2022 (stable) |
| Module | ESNext | preserve (bundler-friendly) |
| Strictness | `strict` only | + `noUncheckedIndexedAccess`, `noImplicitOverride`, `exactOptionalPropertyTypes` |
| Side Effects | ❌ | ✅ `noUncheckedSideEffectImports` |
| Noisy Rules | ✅ (noUnusedLocals, etc.) | ❌ (opt-in only) |

---

## 🚀 Why These Settings?

### `noUncheckedIndexedAccess`
**Problem:** Array/object access can return `undefined` at runtime
```typescript
const users = ['Alice', 'Bob'];
const user = users[5]; // Runtime: undefined, Type: string ❌
```

**Solution:** Forces you to check
```typescript
const user = users[5]; // Type: string | undefined ✅
if (user) {
  console.log(user.toUpperCase());
}
```

### `exactOptionalPropertyTypes`
**Problem:** `undefined` vs missing property confusion
```typescript
type Config = { port?: number };
const config: Config = { port: undefined }; // Should this be allowed?
```

**Solution:** Enforces semantic correctness
```typescript
const config1: Config = {}; // ✅ Property omitted
const config2: Config = { port: 3000 }; // ✅ Property provided
const config3: Config = { port: undefined }; // ❌ Error!
```

### `verbatimModuleSyntax`
**Problem:** Ambiguous imports can cause issues
```typescript
import { User } from './types'; // Is this a type or value?
```

**Solution:** Explicit type imports
```typescript
import type { User } from './types'; // Clearly a type
import { getUser } from './api';     // Clearly a value
```

---

## 📚 Resources

- [Total TypeScript TSConfig Cheat Sheet](https://www.totaltypescript.com/tsconfig-cheat-sheet)
- [TypeScript 5.9 Release Notes](https://www.typescriptlang.org/docs/handbook/release-notes/typescript-5-9.html)
- [TSConfig Reference](https://www.typescriptlang.org/tsconfig/)
- [TypeScript Deep Dive](https://basarat.gitbook.io/typescript/)

---

## 🔄 Version History

### v1.0.0 (2025-11-11)
- ✅ Complete rewrite based on TS 5.9+ best practices
- ✅ Added `noUncheckedIndexedAccess`, `noImplicitOverride`, `exactOptionalPropertyTypes`
- ✅ Added `noUncheckedSideEffectImports` (TS 5.9+)
- ✅ Changed `module: preserve` for better bundler compatibility
- ✅ Removed noisy rules (`noUnusedLocals`, `noUnusedParameters`, etc.)
- ✅ Added `nextjs.json`, `library.json`, `library-react.json` configs
- ✅ Comprehensive documentation

### v0.0.0 (Previous)
- Basic configs with ES2022 target
- Included noisy strictness rules
- Less comprehensive

---

**Created by:** TEAM-471  
**Last Updated:** 2025-11-11
