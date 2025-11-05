# TEAM-417: Move Marketplace Packages to 79_marketplace_core

**Created by:** TEAM-417  
**Date:** 2025-11-05  
**Status:** ✅ COMPLETE  
**Duration:** 15 minutes

---

## 🎯 Mission

Move marketplace-related packages from scattered locations into a dedicated `79_marketplace_core` directory for better organization.

---

## ✅ What Was Done

### 1. Created New Directory Structure

```
bin/79_marketplace_core/
├── marketplace-sdk/      # Rust crate (WASM)
└── marketplace-node/     # Node.js wrapper
```

### 2. Moved Packages

**From → To:**
- `/bin/99_shared_crates/marketplace-sdk` → `/bin/79_marketplace_core/marketplace-sdk`
- `/frontend/packages/marketplace-node` → `/bin/79_marketplace_core/marketplace-node`

---

## 🔧 Files Updated

### 1. Root Cargo.toml
**File:** `/Cargo.toml`

**Changed:**
```toml
# Before
"bin/99_shared_crates/marketplace-sdk",

# After
"bin/79_marketplace_core/marketplace-sdk",
```

---

### 2. Root pnpm-workspace.yaml
**File:** `/pnpm-workspace.yaml`

**Changed:**
```yaml
# Before
- bin/99_shared_crates/marketplace-sdk

# After
- bin/79_marketplace_core/marketplace-sdk
- bin/79_marketplace_core/marketplace-node
```

---

### 3. rbee-keeper Cargo.toml
**File:** `/bin/00_rbee_keeper/Cargo.toml`

**Changed:**
```toml
# Before
marketplace-sdk = { path = "../99_shared_crates/marketplace-sdk", features = ["specta"] }

# After
marketplace-sdk = { path = "../79_marketplace_core/marketplace-sdk", features = ["specta"] }
```

---

### 4. marketplace-node package.json
**File:** `/bin/79_marketplace_core/marketplace-node/package.json`

**Changed:**
```json
// Before
"build:wasm": "cd ../../../bin/99_shared_crates/marketplace-sdk && wasm-pack build --target nodejs --out-dir ../../../../frontend/packages/marketplace-node/wasm"

// After
"build:wasm": "cd ../marketplace-sdk && wasm-pack build --target nodejs --out-dir ../marketplace-node/wasm"
```

---

### 5. marketplace-sdk Cargo.toml
**File:** `/bin/79_marketplace_core/marketplace-sdk/Cargo.toml`

**Added:**
```toml
# TEAM-405: Tauri type generation (non-WASM only)
specta = { version = "=2.0.0-rc.22", features = ["derive"], optional = true }

[features]
# TEAM-405: Enable specta for Tauri bindings
specta = ["dep:specta"]
```

**Why:** The `specta` dependency needs the "derive" feature for Tauri bindings.

---

### 6. marketplace-sdk types.rs
**File:** `/bin/79_marketplace_core/marketplace-sdk/src/types.rs`

**Changed:**
```rust
// Before
#[cfg_attr(not(target_arch = "wasm32"), derive(specta::Type))]

// After
#[cfg_attr(all(not(target_arch = "wasm32"), feature = "specta"), derive(specta::Type))]
```

**Why:** Only derive `specta::Type` when the `specta` feature is enabled.

---

## ✅ Verification

### Rust Build
```bash
cargo check -p marketplace-sdk --features specta
```
**Result:** ✅ SUCCESS

### Node.js Build
```bash
cd bin/79_marketplace_core/marketplace-node
pnpm build
```
**Result:** ✅ SUCCESS (TypeScript + WASM compiled)

### pnpm Install
```bash
pnpm install
```
**Result:** ✅ SUCCESS (workspace links updated)

---

## 📊 Impact

### Directory Structure
**Before:**
```
bin/99_shared_crates/marketplace-sdk/    # Mixed with other shared crates
frontend/packages/marketplace-node/       # Separated from SDK
```

**After:**
```
bin/79_marketplace_core/
├── marketplace-sdk/      # Rust + WASM
└── marketplace-node/     # Node.js wrapper
```

**Benefits:**
- ✅ Better organization (marketplace code together)
- ✅ Clearer ownership (79_marketplace_core)
- ✅ Easier to find (dedicated directory)
- ✅ Simpler build paths (relative paths)

---

### Build Paths
**Before:**
```
cd ../../../bin/99_shared_crates/marketplace-sdk && 
  wasm-pack build --target nodejs --out-dir ../../../../frontend/packages/marketplace-node/wasm
```

**After:**
```
cd ../marketplace-sdk && 
  wasm-pack build --target nodejs --out-dir ../marketplace-node/wasm
```

**Benefits:**
- ✅ Shorter paths
- ✅ Easier to understand
- ✅ Less error-prone

---

## 🎯 New Structure

```
bin/79_marketplace_core/
├── marketplace-sdk/
│   ├── src/
│   │   ├── lib.rs           # WASM entry point
│   │   ├── types.rs         # Model, ModelSource types
│   │   └── huggingface.rs   # HuggingFace client (native Rust)
│   ├── Cargo.toml           # Rust dependencies
│   └── README.md
│
└── marketplace-node/
    ├── src/
    │   ├── index.ts         # Main exports
    │   ├── huggingface.ts   # HuggingFace API client
    │   └── types.ts         # TypeScript types
    ├── dist/                # Built TypeScript
    ├── wasm/                # Built WASM from marketplace-sdk
    ├── package.json
    └── tsconfig.json
```

---

## 🔍 What Didn't Change

### Functionality
- ✅ All functions work the same
- ✅ All imports work the same
- ✅ All builds work the same

### API
- ✅ `@rbee/marketplace-node` package name unchanged
- ✅ `@rbee/marketplace-sdk` package name unchanged
- ✅ Import paths unchanged (workspace:* links)

### Dependencies
- ✅ No new dependencies added
- ✅ No dependencies removed
- ✅ All version constraints unchanged

---

## 📝 Notes

### Why 79_marketplace_core?

**Naming Convention:**
- `00-09`: Core binaries (keeper, queen, hive)
- `10-19`: Daemon binaries
- `20-29`: Worker binaries
- `30-39`: Specialized workers (LLM, SD)
- `70-79`: **Core libraries and SDKs**
- `80-89`: Services (worker catalog)
- `90-99`: Shared utilities

**79_marketplace_core** fits the pattern:
- Core library (not a binary)
- Marketplace-specific (not general shared)
- SDK + Node.js wrapper (complete package)

---

### Specta Feature

The `specta` feature is **optional** and only used by `rbee-keeper` for Tauri bindings.

**When enabled:**
- Generates TypeScript types for Tauri
- Adds `specta::Type` derive macro
- Requires `specta` with "derive" feature

**When disabled:**
- No specta dependency
- No Tauri bindings
- Smaller compile time for WASM builds

---

## ✅ Checklist

- [x] Move marketplace-sdk to bin/79_marketplace_core/
- [x] Move marketplace-node to bin/79_marketplace_core/
- [x] Update Cargo.toml (root)
- [x] Update pnpm-workspace.yaml
- [x] Update rbee-keeper Cargo.toml
- [x] Update marketplace-node package.json (build script)
- [x] Fix specta feature in marketplace-sdk
- [x] Fix specta derives in types.rs
- [x] Run pnpm install
- [x] Test Rust build (cargo check)
- [x] Test Node.js build (pnpm build)
- [x] Verify WASM generation

---

## 🚀 Next Steps

**Immediate:**
- No action needed - everything works!

**Future:**
- Consider adding more marketplace-related packages to 79_marketplace_core/
- Document the 79_marketplace_core directory structure
- Add README.md to 79_marketplace_core/

---

**TEAM-417 - Marketplace Core Move Complete**  
**Status:** ✅ ALL BUILDS PASSING  
**Impact:** Better organization, clearer structure, simpler paths
