# TEAM-423: ROCm Support Added

**Date:** 2025-11-08  
**Issue:** Rust WorkerType enum missing ROCm variant  
**Status:** ✅ FIXED

---

## 🐛 Problem

The Rust `WorkerType` enum in `artifacts-contract` only supported `cpu`, `cuda`, and `metal`, but:
- TypeScript types already included `rocm` (line 14 in `types.ts`)
- Next.js filters included ROCm option
- GUI filters included ROCm option

This caused a deserialization error when the Hono worker catalog returned workers with `workerType: "rocm"`:

```
Error: Failed to parse worker catalog response: 
unknown variant `rocm`, expected one of `cpu`, `cuda`, `metal`
```

---

## ✅ Solution

Added ROCm support to the Rust contract (canonical source of truth):

### 1. Updated `WorkerType` Enum
**File:** `bin/97_contracts/artifacts-contract/src/worker.rs`

```rust
pub enum WorkerType {
    Cpu,
    Cuda,
    Metal,
    Rocm,  // ← ADDED
}
```

### 2. Updated `binary_name()` Method
```rust
pub fn binary_name(&self) -> &str {
    match self {
        WorkerType::Cpu => "llm-worker-rbee-cpu",
        WorkerType::Cuda => "llm-worker-rbee-cuda",
        WorkerType::Metal => "llm-worker-rbee-metal",
        WorkerType::Rocm => "llm-worker-rbee-rocm",  // ← ADDED
    }
}
```

### 3. Updated `build_features()` Method
```rust
pub fn build_features(&self) -> Option<&str> {
    match self {
        WorkerType::Cpu => Some("cpu"),
        WorkerType::Cuda => Some("cuda"),
        WorkerType::Metal => Some("metal"),
        WorkerType::Rocm => Some("rocm"),  // ← ADDED
    }
}
```

### 4. Updated `worker_type_to_string()` in marketplace-sdk
**File:** `bin/79_marketplace_core/marketplace-sdk/src/lib.rs`

```rust
pub fn worker_type_to_string(worker_type: WorkerType) -> String {
    match worker_type {
        WorkerType::Cpu => "cpu".to_string(),
        WorkerType::Cuda => "cuda".to_string(),
        WorkerType::Metal => "metal".to_string(),
        WorkerType::Rocm => "rocm".to_string(),  // ← ADDED
    }
}
```

---

## 🎯 Now Consistent Across Stack

### ✅ Rust Contract (Canonical Source)
```rust
// bin/97_contracts/artifacts-contract/src/worker.rs
pub enum WorkerType {
    Cpu, Cuda, Metal, Rocm  // ✅ All 4 variants
}
```

### ✅ TypeScript Types
```typescript
// bin/80-hono-worker-catalog/src/types.ts
export type WorkerType = "cpu" | "cuda" | "metal" | "rocm";  // ✅ All 4
```

### ✅ Next.js Filters
```typescript
// frontend/apps/marketplace/app/workers/filters.ts
backend: 'all' | 'cpu' | 'cuda' | 'metal' | 'rocm'  // ✅ All 4
```

### ✅ GUI Filters
```typescript
// bin/00_rbee_keeper/ui/src/pages/MarketplaceRbeeWorkers.tsx
backend: 'all' | 'cpu' | 'cuda' | 'metal' | 'rocm'  // ✅ All 4
```

---

## 📊 Parity Achieved

| Layer | CPU | CUDA | Metal | ROCm | Status |
|-------|-----|------|-------|------|--------|
| **Rust Contract** | ✅ | ✅ | ✅ | ✅ | Complete |
| **marketplace-sdk** | ✅ | ✅ | ✅ | ✅ | Complete |
| **TypeScript Types** | ✅ | ✅ | ✅ | ✅ | Complete |
| **Next.js Filters** | ✅ | ✅ | ✅ | ✅ | Complete |
| **GUI Filters** | ✅ | ✅ | ✅ | ✅ | Complete |

---

## 🔍 Why This Happened

The TypeScript types and filters were added first (TEAM-461 comment in `types.ts`), but the Rust contract wasn't updated at the same time. This created a mismatch where:
- Frontend expected 4 worker types
- Backend only supported 3 worker types
- Deserialization failed when ROCm workers appeared

---

## ✅ Verification

### Build Status
```bash
cargo check -p artifacts-contract
✓ Finished `dev` profile

cargo check -p marketplace-sdk
✓ Finished `dev` profile

cargo build --bin rbee-keeper
✓ Finished `dev` profile
```

### Expected Behavior
1. **Hono catalog** can return workers with `workerType: "rocm"`
2. **Rust deserializes** ROCm workers successfully
3. **GUI filters** show ROCm option
4. **Next.js filters** show ROCm option
5. **All filters work** correctly with ROCm workers

---

## 📝 Files Modified

```
modified:   bin/97_contracts/artifacts-contract/src/worker.rs
modified:   bin/79_marketplace_core/marketplace-sdk/src/lib.rs
```

**Changes:**
- Added `Rocm` variant to `WorkerType` enum
- Added ROCm cases to all match statements
- Added documentation comments

---

## 🎯 Design Principle

**Canonical Source of Truth:** The Rust contract (`artifacts-contract`) is the single source of truth for worker types. TypeScript types are generated from it via tsify/WASM.

**When adding new worker types:**
1. ✅ Add to Rust enum first (`artifacts-contract`)
2. ✅ Update all match statements
3. ✅ TypeScript types auto-generate via tsify
4. ✅ Update filters in Next.js and GUI

This ensures parity across the entire stack.

---

## ✅ Result

ROCm workers are now fully supported:
- ✅ Rust can deserialize ROCm workers
- ✅ TypeScript types include ROCm
- ✅ Next.js filters include ROCm
- ✅ GUI filters include ROCm
- ✅ All layers have parity

**Status:** ✅ COMPLETE

---

**TEAM-423 Sign-off:** ROCm support added to Rust contract and all match statements. Full parity achieved across Rust, TypeScript, Next.js, and GUI.
