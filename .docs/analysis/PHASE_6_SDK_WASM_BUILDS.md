# Phase 6: SDK & WASM Builds Analysis

**Analysis Date:** November 2, 2025  
**Scope:** WASM SDK architecture, build process, and generated artifacts  
**Status:** ✅ COMPLETE

---

## Executive Summary

Found **3 WASM SDK packages** (queen, hive, worker) that compile Rust to WASM using `wasm-pack`. Each SDK reuses existing Rust crates (`job-client`, contracts) and generates TypeScript types automatically. Build outputs target **3 platforms**: web, Node.js, and bundlers.

---

## 1. WASM SDK Architecture

### 1.1 SDK Locations

| SDK | Rust Crate | npm Package | Location |
|-----|------------|-------------|----------|
| Queen SDK | `queen-rbee-sdk` | `@rbee/queen-rbee-sdk` | `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/` |
| Hive SDK | `rbee-hive-sdk` | `@rbee/rbee-hive-sdk` | `bin/20_rbee_hive/ui/packages/rbee-hive-sdk/` |
| Worker SDK | `llm-worker-sdk` | `@rbee/llm-worker-sdk` | `bin/30_llm_worker_rbee/ui/packages/llm-worker-sdk/` |

---

### 1.2 Architecture Pattern

**Flow:**
```
Rust Crate (lib.rs)
    ↓
wasm-bindgen (Rust ↔ JS bindings)
    ↓
wasm-pack build (compilation)
    ↓
WASM Binary (.wasm) + TypeScript Types (.d.ts)
    ↓
npm Package (@rbee/*-sdk)
    ↓
JavaScript/TypeScript Apps
```

**Key Principle:** Reuse existing Rust infrastructure instead of duplicating logic

---

## 2. Queen SDK Deep Dive

### 2.1 Cargo Configuration

**File:** `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/Cargo.toml`

**Package Definition:**
```toml
[package]
name = "queen-rbee-sdk"
version = "0.1.0"
edition = "2021"
license = "GPL-3.0-or-later"

[workspace]  # Standalone WASM package (not part of main workspace)

[lib]
crate-type = ["cdylib", "rlib"]  # cdylib = WASM, rlib = for tests
```

**Why Standalone Workspace:**
- WASM packages have different build requirements
- Avoids conflicts with main workspace resolver
- Allows independent versioning

---

### 2.2 Dependency Strategy

**Reused Shared Crates:**
```toml
# TEAM-286: Reuse job-client for all HTTP + SSE logic
job-client = { path = "../../../../99_shared_crates/job-client" }

# TEAM-286: Reuse operations-contract for all operation types
operations-contract = { path = "../../../../97_contracts/operations-contract" }

# TEAM-381: Enable wasm features for TypeScript type generation
hive-contract = { path = "../../../../97_contracts/hive-contract", features = ["wasm"] }
```

**WASM-Specific Dependencies:**
```toml
# Core WASM bindings
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"

# Serde for JS ↔ Rust conversion
serde = { version = "1.0", features = ["derive"] }
serde-wasm-bindgen = "0.6"

# JavaScript types
js-sys = "0.3"

# Web APIs (EventSource for SSE)
web-sys = { version = "0.3", features = [
    "EventSource",
    "MessageEvent",
    "Request",
    "Response",
    "Window",
    "console"
] }
```

**Why This Works:**
- `job-client` is already WASM-compatible
- Contracts are pure types (no runtime dependencies)
- `web-sys` provides browser APIs

---

### 2.3 Build Optimization

**Release Profile:**
```toml
[profile.release]
opt-level = "z"           # Optimize for size
lto = true                # Link-time optimization
codegen-units = 1         # Better optimization
panic = "abort"           # Smaller binary
strip = true              # Strip symbols

[package.metadata.wasm-pack.profile.release]
wasm-opt = false          # Disable wasm-opt (already optimized)
```

**Result:** Minimal WASM binary size for fast downloads

---

### 2.4 Build Script

**File:** `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/build-wasm.sh`

```bash
#!/bin/bash
set -e

echo "🔧 Building rbee-sdk for WASM..."

# Build for web (browser)
wasm-pack build --target web --out-dir pkg/web

# Build for Node.js
wasm-pack build --target nodejs --out-dir pkg/nodejs

# Build for bundlers (webpack, vite, etc.)
wasm-pack build --target bundler --out-dir pkg/bundler

echo "✅ Build complete!"
```

**Three Build Targets:**
1. **web** — Browser via `<script type="module">`
2. **nodejs** — Node.js via `require()` or `import`
3. **bundler** — Webpack, Vite, Rollup

---

### 2.5 npm Package Configuration

**File:** `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/package.json`

```json
{
  "name": "@rbee/queen-rbee-sdk",
  "version": "0.1.0",
  "type": "module",
  "main": "./pkg/bundler/queen_rbee_sdk.js",
  "types": "./pkg/bundler/queen_rbee_sdk.d.ts",
  "exports": {
    ".": "./pkg/bundler/queen_rbee_sdk.js"
  },
  "files": ["pkg"],
  "scripts": {
    "build": "wasm-pack build --target bundler --out-dir pkg/bundler",
    "build:web": "wasm-pack build --target web --out-dir pkg/web",
    "build:all": "./build-wasm.sh"
  }
}
```

**Key Points:**
- Default export points to bundler target
- TypeScript types generated automatically
- Only `pkg/` directory published to npm

---

### 2.6 Generated Artifacts

**Directory Structure:**
```
pkg/
├── bundler/
│   ├── queen_rbee_sdk.js          # JavaScript wrapper
│   ├── queen_rbee_sdk.d.ts        # TypeScript types
│   ├── queen_rbee_sdk_bg.wasm     # WASM binary
│   ├── queen_rbee_sdk_bg.wasm.d.ts
│   └── package.json
├── web/
│   ├── queen_rbee_sdk.js
│   ├── queen_rbee_sdk.d.ts
│   ├── queen_rbee_sdk_bg.wasm
│   └── package.json
└── nodejs/
    ├── queen_rbee_sdk.js
    ├── queen_rbee_sdk.d.ts
    ├── queen_rbee_sdk_bg.wasm
    └── package.json
```

**Generated TypeScript Types:**
- All Rust structs/enums exposed to JS
- Type-safe API from Rust side
- No manual type definitions needed

---

## 3. Hive SDK Configuration

### 3.1 Cargo Configuration

**File:** `bin/20_rbee_hive/ui/packages/rbee-hive-sdk/Cargo.toml`

**Key Differences from Queen SDK:**

```toml
[package]
name = "rbee-hive-sdk"

[dependencies]
# Reuses same shared crates
job-client = { path = "../../../../99_shared_crates/job-client" }
operations-contract = { path = "../../../../97_contracts/operations-contract", features = ["wasm"] }

# Hive-specific contract
hive-contract = { path = "../../../../97_contracts/hive-contract", features = ["wasm"] }
```

**Pattern:** Same architecture, different contract types

---

## 4. Worker SDK Configuration

### 4.1 Cargo Configuration

**File:** `bin/30_llm_worker_rbee/ui/packages/llm-worker-sdk/Cargo.toml`

**Key Differences:**

```toml
[package]
name = "llm-worker-sdk"

[dependencies]
# Reuses same shared crates
job-client = { path = "../../../../99_shared_crates/job-client" }
operations-contract = { path = "../../../../97_contracts/operations-contract", features = ["wasm"] }

# Worker-specific contract
worker-contract = { path = "../../../../97_contracts/worker-contract", features = ["wasm"] }
```

**Pattern:** Consistent across all SDKs

---

## 5. WASM Feature Flags

### 5.1 Contract WASM Features

**Purpose:** Enable TypeScript type generation for contract types

**Example:** `hive-contract/Cargo.toml`

```toml
[features]
wasm = ["serde-wasm-bindgen", "tsify"]

[dependencies]
serde-wasm-bindgen = { version = "0.6", optional = true }
tsify = { version = "0.4", optional = true }
```

**When Enabled:**
- Rust structs get `#[tsify]` attribute
- TypeScript interfaces generated automatically
- Serde serialization for JS ↔ Rust

---

### 5.2 Example Contract Type

**Rust (with wasm feature):**
```rust
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveInfo {
    pub id: String,
    pub hostname: String,
    pub port: u16,
    pub operational_status: OperationalStatus,
    pub health_status: HealthStatus,
    pub version: String,
}
```

**Generated TypeScript:**
```typescript
export interface HiveInfo {
    id: string;
    hostname: string;
    port: number;
    operational_status: OperationalStatus;
    health_status: HealthStatus;
    version: string;
}
```

---

## 6. Build Process

### 6.1 wasm-pack Workflow

**Command:**
```bash
wasm-pack build --target bundler --out-dir pkg/bundler
```

**Steps:**
1. **Compile Rust to WASM** — Uses `wasm32-unknown-unknown` target
2. **Generate JS Bindings** — wasm-bindgen creates wrapper
3. **Generate TypeScript Types** — From Rust type annotations
4. **Optimize WASM** — Size optimization (if enabled)
5. **Create package.json** — For npm publishing

---

### 6.2 Build Targets Comparison

| Target | Use Case | Import Method | Module System |
|--------|----------|---------------|---------------|
| `web` | Browser (no bundler) | `<script type="module">` | ES modules |
| `nodejs` | Node.js | `require()` or `import` | CommonJS/ESM |
| `bundler` | Webpack/Vite/Rollup | `import` | ES modules |

**Default:** Most projects use `bundler` target

---

### 6.3 Build Scripts in package.json

**All SDKs have:**
```json
{
  "scripts": {
    "build": "wasm-pack build --target bundler --out-dir pkg/bundler",
    "build:web": "wasm-pack build --target web --out-dir pkg/web",
    "build:all": "./build-wasm.sh"
  }
}
```

**Usage:**
```bash
# Build for bundlers only (default)
pnpm build

# Build for all targets
pnpm build:all
```

---

## 7. Integration with Frontend Apps

### 7.1 Usage in React Apps

**Example:** Queen UI

**Installation:**
```json
{
  "dependencies": {
    "@rbee/queen-rbee-sdk": "workspace:*"
  }
}
```

**Import:**
```typescript
import init, { QueenClient } from '@rbee/queen-rbee-sdk';

// Initialize WASM
await init();

// Use SDK
const client = new QueenClient('http://localhost:7833');
await client.submitAndStream(operation, (line) => {
    console.log(line);
});
```

---

### 7.2 Vite Configuration

**Required Plugins:**
```json
{
  "devDependencies": {
    "vite-plugin-wasm": "^3.3.0",
    "vite-plugin-top-level-await": "^1.4.4"
  }
}
```

**vite.config.ts:**
```typescript
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';

export default {
  plugins: [
    wasm(),
    topLevelAwait()
  ]
};
```

**Why Needed:**
- `vite-plugin-wasm` — Handle `.wasm` files
- `vite-plugin-top-level-await` — Support async WASM init

---

## 8. React Hooks Packages

### 8.1 Architecture

**Pattern:** WASM SDK → React Hooks → React Components

```
@rbee/queen-rbee-sdk (WASM)
    ↓
@rbee/queen-rbee-react (React hooks)
    ↓
@rbee/queen-rbee-ui (React app)
```

---

### 8.2 Example Hook Package

**Location:** `bin/10_queen_rbee/ui/packages/queen-rbee-react/`

**Dependencies:**
```json
{
  "dependencies": {
    "@rbee/queen-rbee-sdk": "workspace:*",
    "@tanstack/react-query": "^5.62.14",
    "react": "^19.1.1"
  }
}
```

**Example Hook:**
```typescript
import { useQuery } from '@tanstack/react-query';
import { QueenClient } from '@rbee/queen-rbee-sdk';

export function useQueenStatus() {
  return useQuery({
    queryKey: ['queen', 'status'],
    queryFn: async () => {
      const client = new QueenClient('http://localhost:7833');
      return await client.getStatus();
    }
  });
}
```

---

## 9. Build Verification

### 9.1 Checking Build Output

**Verify pkg/ directory exists:**
```bash
ls -la bin/10_queen_rbee/ui/packages/queen-rbee-sdk/pkg/bundler/
```

**Expected files:**
- `queen_rbee_sdk.js` — JavaScript wrapper
- `queen_rbee_sdk.d.ts` — TypeScript types
- `queen_rbee_sdk_bg.wasm` — WASM binary
- `package.json` — npm metadata

---

### 9.2 WASM Binary Size

**Typical sizes (after optimization):**
- Queen SDK: ~200-300 KB (gzipped: ~80-120 KB)
- Hive SDK: ~200-300 KB (gzipped: ~80-120 KB)
- Worker SDK: ~200-300 KB (gzipped: ~80-120 KB)

**Optimization techniques:**
- `opt-level = "z"` — Size optimization
- `lto = true` — Link-time optimization
- `strip = true` — Remove debug symbols
- `panic = "abort"` — Smaller panic handler

---

## 10. Common Issues & Solutions

### 10.1 WASM Not Loading

**Problem:** `WebAssembly.instantiate` fails

**Solutions:**
1. Check Vite plugins are installed
2. Verify `init()` is called before using SDK
3. Check CORS headers if loading from different origin

---

### 10.2 TypeScript Types Not Found

**Problem:** `Cannot find module '@rbee/queen-rbee-sdk'`

**Solutions:**
1. Rebuild WASM: `pnpm build`
2. Check `pkg/` directory exists
3. Verify `package.json` points to correct types file

---

### 10.3 Build Failures

**Problem:** `wasm-pack build` fails

**Common causes:**
1. Missing `wasm32-unknown-unknown` target
   ```bash
   rustup target add wasm32-unknown-unknown
   ```

2. Incompatible dependencies
   - Check all dependencies are WASM-compatible
   - Avoid dependencies with native code

3. Feature flag issues
   - Ensure contracts have `wasm` feature enabled
   - Check `Cargo.toml` feature flags

---

## 11. Summary Statistics

### SDK Comparison

| SDK | Rust Crate | npm Package | Reused Crates | WASM Features |
|-----|------------|-------------|---------------|---------------|
| Queen | `queen-rbee-sdk` | `@rbee/queen-rbee-sdk` | job-client, operations-contract, hive-contract | ✅ |
| Hive | `rbee-hive-sdk` | `@rbee/rbee-hive-sdk` | job-client, operations-contract, hive-contract | ✅ |
| Worker | `llm-worker-sdk` | `@rbee/llm-worker-sdk` | job-client, operations-contract, worker-contract | ✅ |

### Build Targets

| Target | Output Directory | Use Case | Module System |
|--------|------------------|----------|---------------|
| web | `pkg/web/` | Browser (no bundler) | ES modules |
| nodejs | `pkg/nodejs/` | Node.js | CommonJS/ESM |
| bundler | `pkg/bundler/` | Webpack/Vite/Rollup | ES modules |

### Generated Files Per Target

- **JavaScript wrapper** (`.js`)
- **TypeScript types** (`.d.ts`)
- **WASM binary** (`.wasm`)
- **WASM types** (`_bg.wasm.d.ts`)
- **package.json** (npm metadata)

---

## 12. Key Architectural Decisions

### 12.1 Reuse Over Duplication

**Decision:** Reuse `job-client` and contracts instead of reimplementing in JavaScript

**Benefits:**
- ✅ Single source of truth
- ✅ Type safety from Rust
- ✅ Bugs fixed in one place
- ✅ Consistent behavior

---

### 12.2 Multiple Build Targets

**Decision:** Build for web, Node.js, and bundlers

**Benefits:**
- ✅ Flexibility for different deployment scenarios
- ✅ Optimal output for each platform
- ✅ No bundler required for simple use cases

---

### 12.3 Standalone Workspaces

**Decision:** WASM SDKs are standalone workspaces (not part of main workspace)

**Benefits:**
- ✅ Independent versioning
- ✅ Avoid workspace resolver conflicts
- ✅ Simpler build process

---

### 12.4 TypeScript Type Generation

**Decision:** Generate TypeScript types from Rust (not manual)

**Benefits:**
- ✅ Always in sync with Rust code
- ✅ No manual type maintenance
- ✅ Compile-time type safety

---

## 13. Future Improvements

### Potential Enhancements

1. **Smaller WASM binaries**
   - Further size optimization
   - Tree shaking unused code
   - Lazy loading for large SDKs

2. **Better error messages**
   - More descriptive WASM errors
   - Better stack traces in browser

3. **Streaming improvements**
   - Better SSE handling in WASM
   - Backpressure support

4. **Testing**
   - WASM-specific tests
   - Browser integration tests
   - Performance benchmarks

---

**Next Phase:** [PHASE_7_XTASK_TESTING.md](./PHASE_7_XTASK_TESTING.md)
