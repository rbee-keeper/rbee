# Artifacts Contract Migration Plan

**Date:** 2025-11-04  
**Team:** TEAM-402  
**Status:** 📋 MIGRATION PLAN

---

## 🎯 The Problem

**Current State (WRONG):**
```
model-catalog/src/types.rs
    ↓
Defines ModelEntry

worker-catalog/src/types.rs
    ↓
Defines WorkerBinary

model-provisioner
    ↓
Depends on model-catalog (for ModelEntry)
    ↓
CIRCULAR DEPENDENCY if catalog depends on provisioner!
```

**Who Needs These Types:**
1. ✅ **model-catalog** - Storage
2. ✅ **model-provisioner** - Provisioning
3. ✅ **worker-catalog** - Storage
4. ✅ **worker-provisioner** - Provisioning
5. ✅ **rbee-hive** - Operations
6. ✅ **marketplace-sdk** - Display
7. ✅ **keeper UI** - Display

---

## ✅ The Solution: artifacts-contract

Create a **pure types crate** in `/home/vince/Projects/llama-orch/bin/97_contracts/artifacts-contract/`

### What Goes In artifacts-contract

```rust
// bin/97_contracts/artifacts-contract/src/lib.rs

/// Core artifact types shared across:
/// - Catalogs (model-catalog, worker-catalog)
/// - Provisioners (model-provisioner, worker-provisioner)
/// - Marketplace (marketplace-sdk)
/// - UI (keeper, marketplace-site)

pub mod model;
pub mod worker;
pub mod status;

pub use model::ModelEntry;
pub use worker::WorkerBinary;
pub use status::ArtifactStatus;
```

### Dependencies

```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
chrono = { version = "0.4", features = ["serde"] }
tsify = "0.4"  # For TypeScript generation
wasm-bindgen = "0.2"  # For WASM compatibility
```

**NO OTHER DEPENDENCIES!** Pure types only.

---

## 🔄 Migration Steps

### Step 1: Create artifacts-contract

```bash
cd bin/97_contracts
mkdir -p artifacts-contract/src
```

**Files to create:**
- `Cargo.toml`
- `src/lib.rs`
- `src/model.rs` (move ModelEntry from model-catalog)
- `src/worker.rs` (move WorkerBinary from worker-catalog)
- `src/status.rs` (move ArtifactStatus from artifact-catalog)
- `README.md`

### Step 2: Move ModelEntry

**From:** `bin/25_rbee_hive_crates/model-catalog/src/types.rs`  
**To:** `bin/97_contracts/artifacts-contract/src/model.rs`

```rust
// bin/97_contracts/artifacts-contract/src/model.rs
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tsify::Tsify;

/// Model entry - shared across catalog, provisioner, marketplace
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    pub path: PathBuf,
    pub size: u64,
    pub status: super::status::ArtifactStatus,
    pub added_at: chrono::DateTime<chrono::Utc>,
}
```

### Step 3: Move WorkerBinary

**From:** `bin/25_rbee_hive_crates/worker-catalog/src/types.rs`  
**To:** `bin/97_contracts/artifacts-contract/src/worker.rs`

```rust
// bin/97_contracts/artifacts-contract/src/worker.rs
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tsify::Tsify;

/// Worker binary - shared across catalog, provisioner, marketplace
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct WorkerBinary {
    pub id: String,
    pub worker_type: WorkerType,
    pub platform: Platform,
    pub path: PathBuf,
    pub size: u64,
    pub status: super::status::ArtifactStatus,
    pub version: String,
    pub added_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Tsify, PartialEq, Eq)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub enum WorkerType {
    CpuLlm,
    CudaLlm,
    MetalLlm,
}

#[derive(Debug, Clone, Serialize, Deserialize, Tsify, PartialEq, Eq)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub enum Platform {
    Linux,
    MacOS,
    Windows,
}
```

### Step 4: Move ArtifactStatus

**From:** `bin/25_rbee_hive_crates/artifact-catalog/src/types.rs`  
**To:** `bin/97_contracts/artifacts-contract/src/status.rs`

```rust
// bin/97_contracts/artifacts-contract/src/status.rs
use serde::{Deserialize, Serialize};
use tsify::Tsify;

/// Artifact status - shared across all artifacts
#[derive(Debug, Clone, Serialize, Deserialize, Tsify, PartialEq, Eq)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub enum ArtifactStatus {
    Available,
    Downloading,
    Failed,
}
```

### Step 5: Update Dependencies

**Add artifacts-contract to workspace:**
```toml
# /home/vince/Projects/llama-orch/Cargo.toml
[workspace]
members = [
    # ... existing ...
    "bin/97_contracts/artifacts-contract",
]
```

**Update all consumers:**

1. **model-catalog/Cargo.toml:**
   ```toml
   [dependencies]
   artifacts-contract = { path = "../../97_contracts/artifacts-contract" }
   # Remove: rbee-hive-artifact-catalog (for types)
   ```

2. **model-provisioner/Cargo.toml:**
   ```toml
   [dependencies]
   artifacts-contract = { path = "../../97_contracts/artifacts-contract" }
   # Remove: rbee-hive-model-catalog (for types)
   ```

3. **worker-catalog/Cargo.toml:**
   ```toml
   [dependencies]
   artifacts-contract = { path = "../../97_contracts/artifacts-contract" }
   ```

4. **worker-provisioner/Cargo.toml:**
   ```toml
   [dependencies]
   artifacts-contract = { path = "../../97_contracts/artifacts-contract" }
   # Remove: rbee-hive-worker-catalog (for types)
   ```

5. **marketplace-sdk/Cargo.toml:**
   ```toml
   [dependencies]
   artifacts-contract = { path = "../../97_contracts/artifacts-contract" }
   ```

### Step 6: Update Imports

**In all files, change:**
```rust
// OLD
use rbee_hive_model_catalog::ModelEntry;
use rbee_hive_worker_catalog::WorkerBinary;

// NEW
use artifacts_contract::{ModelEntry, WorkerBinary, ArtifactStatus};
```

---

## 📊 New Dependency Graph

```
artifacts-contract (pure types)
    ↓
├── model-catalog (storage)
├── model-provisioner (provisioning)
├── worker-catalog (storage)
├── worker-provisioner (provisioning)
├── marketplace-sdk (display)
└── rbee-hive (operations)
```

**NO CIRCULAR DEPENDENCIES!** ✅

---

## ✅ Benefits

1. **Single Source of Truth**
   - One place for ModelEntry
   - One place for WorkerBinary
   - One place for ArtifactStatus

2. **No Circular Dependencies**
   - Provisioners don't depend on catalogs
   - Catalogs don't depend on provisioners
   - Both depend on artifacts-contract

3. **Marketplace Integration**
   - marketplace-sdk can use same types
   - TypeScript types auto-generated via tsify
   - WASM-compatible

4. **Clean Architecture**
   - Contracts = pure types
   - Catalogs = storage
   - Provisioners = downloading/building
   - rbee-hive = orchestration

---

## 🎯 After Migration: Provisioner Integration

**THEN** we can integrate provisioner into catalog:

```rust
// model-catalog/src/lib.rs
use artifacts_contract::ModelEntry;
use rbee_hive_model_provisioner::ModelProvisioner;  // ✅ No circular dep!

impl ModelCatalog {
    pub async fn provision_and_add(...) -> Result<ModelEntry> {
        let provisioner = ModelProvisioner::new()?;
        let model = provisioner.provision(...).await?;
        self.add(model.clone())?;
        Ok(model)
    }
}
```

**Why this works:**
- `artifacts-contract` has `ModelEntry` (pure type)
- `model-catalog` depends on `artifacts-contract`
- `model-provisioner` depends on `artifacts-contract`
- `model-catalog` can depend on `model-provisioner` ✅
- NO CIRCULAR DEPENDENCY!

---

## 📝 Checklist

- [ ] Create `bin/97_contracts/artifacts-contract/`
- [ ] Move `ModelEntry` to `artifacts-contract/src/model.rs`
- [ ] Move `WorkerBinary` to `artifacts-contract/src/worker.rs`
- [ ] Move `ArtifactStatus` to `artifacts-contract/src/status.rs`
- [ ] Add to workspace
- [ ] Update all Cargo.toml dependencies
- [ ] Update all imports
- [ ] Run `cargo check` on all affected crates
- [ ] Run all tests
- [ ] THEN integrate provisioners into catalogs

---

**TEAM-402 - This is the correct architecture!** ✅

The blocker was that types were in the wrong place. Moving them to `artifacts-contract` solves everything!
