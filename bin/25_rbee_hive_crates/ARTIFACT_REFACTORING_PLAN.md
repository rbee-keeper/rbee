# Artifact System Refactoring Plan

**Date:** 2025-11-04  
**Team:** TEAM-402  
**Status:** 📋 COMPREHENSIVE REFACTORING

---

## 🎯 The Vision: Two-Layer Architecture

### Layer 1: artifacts-contract (Pure Types)
**Location:** `/bin/97_contracts/artifacts-contract/`  
**Purpose:** Pure data types shared across ALL consumers

```
artifacts-contract/
├── src/
│   ├── lib.rs
│   ├── model.rs      (ModelEntry)
│   ├── worker.rs     (WorkerBinary, WorkerType, Platform)
│   └── status.rs     (ArtifactStatus)
└── Cargo.toml
```

**Dependencies:** ONLY `serde`, `chrono`, `tsify`, `wasm-bindgen`  
**Consumers:** model-catalog, model-provisioner, worker-catalog, worker-provisioner, marketplace-sdk, rbee-hive

### Layer 2: artifact-catalog (Reusable Implementations)
**Location:** `/bin/25_rbee_hive_crates/artifact-catalog/`  
**Purpose:** Reusable catalog and provisioner patterns

```
artifact-catalog/
├── src/
│   ├── lib.rs
│   ├── catalog.rs         (FilesystemCatalog<T>, ArtifactCatalog trait)
│   ├── provisioner.rs     (ArtifactProvisioner trait, VendorSource trait)
│   └── types.rs           (Artifact trait - KEEP HERE)
└── Cargo.toml
```

**Dependencies:** `artifacts-contract`, `serde`, `tokio`, `anyhow`  
**Consumers:** model-catalog, model-provisioner, worker-catalog, worker-provisioner

---

## 📊 Current State vs Target State

### Current (WRONG)

```
model-catalog/src/types.rs
    ↓
ModelEntry defined here ❌

worker-catalog/src/types.rs
    ↓
WorkerBinary defined here ❌

artifact-catalog/src/types.rs
    ↓
ArtifactStatus defined here ❌
Artifact trait defined here ✅ (stays)
```

### Target (CORRECT)

```
artifacts-contract/
    ↓
ModelEntry, WorkerBinary, ArtifactStatus ✅

artifact-catalog/
    ↓
Artifact trait ✅
FilesystemCatalog<T> ✅
ArtifactProvisioner trait ✅
VendorSource trait ✅
```

---

## 🔄 Migration Steps

### Phase 1: Create artifacts-contract ✅

**Step 1.1:** Create crate structure
```bash
mkdir -p bin/97_contracts/artifacts-contract/src
```

**Step 1.2:** Create Cargo.toml
```toml
[package]
name = "artifacts-contract"
version = "0.1.0"
edition = "2021"
license = "GPL-3.0-or-later"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
chrono = { version = "0.4", features = ["serde"] }
tsify = "0.4"
wasm-bindgen = "0.2"

[lints]
workspace = true
```

**Step 1.3:** Create lib.rs
```rust
//! Artifact types contract
//!
//! Pure data types for models and workers.
//! Shared across catalogs, provisioners, marketplace, and UI.

#![warn(missing_docs)]

pub mod model;
pub mod worker;
pub mod status;

pub use model::ModelEntry;
pub use worker::{WorkerBinary, WorkerType, Platform};
pub use status::ArtifactStatus;
```

**Step 1.4:** Move types

- Move `ModelEntry` from `model-catalog/src/types.rs` → `artifacts-contract/src/model.rs`
- Move `WorkerBinary`, `WorkerType`, `Platform` from `worker-catalog/src/types.rs` → `artifacts-contract/src/worker.rs`
- Move `ArtifactStatus` from `artifact-catalog/src/types.rs` → `artifacts-contract/src/status.rs`

**Step 1.5:** Add to workspace
```toml
# Cargo.toml (root)
[workspace]
members = [
    # ... existing ...
    "bin/97_contracts/artifacts-contract",
]
```

---

### Phase 2: Update artifact-catalog

**Step 2.1:** Update Cargo.toml
```toml
[dependencies]
artifacts-contract = { path = "../../../97_contracts/artifacts-contract" }
# ... existing deps ...
```

**Step 2.2:** Update types.rs
```rust
// artifact-catalog/src/types.rs

// Re-export from contract
pub use artifacts_contract::{ArtifactStatus, ModelEntry, WorkerBinary};

/// Artifact trait - STAYS HERE (it's behavior, not data)
pub trait Artifact: Send + Sync + Clone {
    fn id(&self) -> &str;
    fn path(&self) -> &std::path::Path;
    fn size(&self) -> u64;
    fn status(&self) -> &ArtifactStatus;
}
```

**Step 2.3:** Update lib.rs
```rust
// artifact-catalog/src/lib.rs
pub use artifacts_contract::{ArtifactStatus, ModelEntry, WorkerBinary};
pub use types::Artifact;
// ... rest stays same ...
```

---

### Phase 3: Update model-catalog

**Step 3.1:** Update Cargo.toml
```toml
[dependencies]
artifacts-contract = { path = "../../../97_contracts/artifacts-contract" }
rbee-hive-artifact-catalog = { path = "../artifact-catalog" }
# ... rest ...
```

**Step 3.2:** Update lib.rs
```rust
// model-catalog/src/lib.rs
pub use artifacts_contract::ModelEntry;

// Remove: mod types;
```

**Step 3.3:** Delete types.rs
```bash
rm bin/25_rbee_hive_crates/model-catalog/src/types.rs
```

---

### Phase 4: Update model-provisioner

**Step 4.1:** Update Cargo.toml
```toml
[dependencies]
artifacts-contract = { path = "../../../97_contracts/artifacts-contract" }
rbee-hive-artifact-catalog = { path = "../artifact-catalog" }
# Remove: rbee-hive-model-catalog (no longer needed for types!)
```

**Step 4.2:** Update imports
```rust
// model-provisioner/src/provisioner.rs
use artifacts_contract::ModelEntry;
use rbee_hive_artifact_catalog::ArtifactProvisioner;
```

---

### Phase 5: Update worker-catalog

**Step 5.1:** Update Cargo.toml
```toml
[dependencies]
artifacts-contract = { path = "../../../97_contracts/artifacts-contract" }
rbee-hive-artifact-catalog = { path = "../artifact-catalog" }
```

**Step 5.2:** Update lib.rs
```rust
// worker-catalog/src/lib.rs
pub use artifacts_contract::{WorkerBinary, WorkerType, Platform};

// Remove: mod types;
```

**Step 5.3:** Delete types.rs
```bash
rm bin/25_rbee_hive_crates/worker-catalog/src/types.rs
```

---

### Phase 6: Update worker-provisioner

**Step 6.1:** Update Cargo.toml
```toml
[dependencies]
artifacts-contract = { path = "../../../97_contracts/artifacts-contract" }
rbee-hive-artifact-catalog = { path = "../artifact-catalog" }
# Remove: rbee-hive-worker-catalog (no longer needed for types!)
```

**Step 6.2:** Update imports
```rust
// worker-provisioner/src/provisioner.rs
use artifacts_contract::{WorkerBinary, WorkerType, Platform};
use rbee_hive_artifact_catalog::ArtifactProvisioner;
```

---

### Phase 7: Update marketplace-sdk

**Step 7.1:** Update Cargo.toml
```toml
[dependencies]
artifacts-contract = { path = "../../97_contracts/artifacts-contract" }
```

**Step 7.2:** Use contract types
```rust
// marketplace-sdk/src/types.rs
pub use artifacts_contract::{ModelEntry, WorkerBinary};

// Can still have marketplace-specific types too
pub struct MarketplaceModel {
    pub entry: ModelEntry,
    pub downloads: u64,
    pub likes: u64,
    // ... marketplace-specific fields ...
}
```

---

### Phase 8: Update rbee-hive

**Step 8.1:** Update Cargo.toml
```toml
[dependencies]
artifacts-contract = { path = "../97_contracts/artifacts-contract" }
rbee-hive-model-catalog = { path = "../25_rbee_hive_crates/model-catalog" }
rbee-hive-worker-catalog = { path = "../25_rbee_hive_crates/worker-catalog" }
```

**Step 8.2:** Update imports
```rust
// rbee-hive/src/handlers/*.rs
use artifacts_contract::{ModelEntry, WorkerBinary};
```

---

## 🎯 Adding CivitAI Support

**Now that types are in the right place, adding CivitAI is clean:**

```rust
// model-provisioner/src/civitai.rs
use artifacts_contract::ModelEntry;
use rbee_hive_artifact_catalog::VendorSource;

pub struct CivitAIVendor {
    // ... implementation ...
}

#[async_trait::async_trait]
impl VendorSource for CivitAIVendor {
    async fn download(
        &self,
        id: &str,
        dest: &Path,
        job_id: &str,
        cancel_token: CancellationToken,
    ) -> Result<u64> {
        // Download from CivitAI
        // Return ModelEntry
    }
    
    fn supports(&self, id: &str) -> bool {
        id.starts_with("civitai:")
    }
    
    fn name(&self) -> &str {
        "CivitAI"
    }
}
```

**Then in ModelProvisioner:**
```rust
// model-provisioner/src/provisioner.rs
use crate::huggingface::HuggingFaceVendor;
use crate::civitai::CivitAIVendor;

pub struct ModelProvisioner {
    vendors: Vec<Box<dyn VendorSource>>,
}

impl ModelProvisioner {
    pub fn new() -> Result<Self> {
        let vendors: Vec<Box<dyn VendorSource>> = vec![
            Box::new(HuggingFaceVendor::new()?),
            Box::new(CivitAIVendor::new()?),
        ];
        Ok(Self { vendors })
    }
}
```

---

## 📊 Final Dependency Graph

```
artifacts-contract (pure types)
    ↓
artifact-catalog (reusable implementations)
    ↓
├── model-catalog (storage)
├── model-provisioner (HuggingFace + CivitAI)
├── worker-catalog (storage)
└── worker-provisioner (PKGBUILD)
    ↓
rbee-hive (orchestration)
    ↓
marketplace-sdk (display)
```

**NO CIRCULAR DEPENDENCIES!** ✅

---

## ✅ Benefits

1. **Pure Types in Contract**
   - ModelEntry, WorkerBinary in one place
   - Shared across ALL consumers
   - WASM-compatible, TypeScript-ready

2. **Reusable Implementations in artifact-catalog**
   - FilesystemCatalog<T> works for any artifact
   - ArtifactProvisioner trait for any vendor
   - VendorSource trait for pluggable vendors

3. **Clean Provisioner Integration**
   - model-catalog can depend on model-provisioner ✅
   - worker-catalog can depend on worker-provisioner ✅
   - No circular dependencies!

4. **Easy to Add Vendors**
   - Add CivitAIVendor to model-provisioner
   - Add GitHubReleaseVendor to worker-provisioner
   - Just implement VendorSource trait

---

## 📝 Checklist

### Phase 1: artifacts-contract
- [ ] Create crate structure
- [ ] Create Cargo.toml
- [ ] Create lib.rs
- [ ] Move ModelEntry to model.rs
- [ ] Move WorkerBinary to worker.rs
- [ ] Move ArtifactStatus to status.rs
- [ ] Add to workspace
- [ ] Compile: `cargo check -p artifacts-contract`

### Phase 2: artifact-catalog
- [ ] Add artifacts-contract dependency
- [ ] Update types.rs (re-export + keep Artifact trait)
- [ ] Update lib.rs
- [ ] Compile: `cargo check -p rbee-hive-artifact-catalog`

### Phase 3-6: Catalogs & Provisioners
- [ ] Update model-catalog
- [ ] Update model-provisioner
- [ ] Update worker-catalog
- [ ] Update worker-provisioner
- [ ] Delete old types.rs files
- [ ] Compile all: `cargo check`

### Phase 7-8: Consumers
- [ ] Update marketplace-sdk
- [ ] Update rbee-hive
- [ ] Run all tests: `cargo test`

### Phase 9: CivitAI Support
- [ ] Add CivitAIVendor to model-provisioner
- [ ] Implement VendorSource trait
- [ ] Add to ModelProvisioner vendors list
- [ ] Test CivitAI downloads

---

**TEAM-402 - Complete Artifact System Refactoring!** ✅

This gives us:
- ✅ Pure types in contracts
- ✅ Reusable implementations in artifact-catalog
- ✅ Clean provisioner integration
- ✅ Easy to add CivitAI and other vendors
- ✅ No circular dependencies
- ✅ Marketplace-ready
