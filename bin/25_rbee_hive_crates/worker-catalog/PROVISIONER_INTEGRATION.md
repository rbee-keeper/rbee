# Worker Catalog + Provisioner Integration

**Date:** 2025-11-04  
**Team:** TEAM-402  
**Status:** 📋 DESIGN

---

## 🎯 Goal

Integrate `worker-provisioner` into `worker-catalog` so that the catalog can provision workers and automatically add them.

---

## 🏗️ Current Architecture (Separated)

```
rbee-hive handler
    ↓
WorkerProvisioner::provision()
    ↓
Returns WorkerBinary
    ↓
rbee-hive manually calls WorkerCatalog::add()
```

**Problems:**
- ❌ Caller has to manually add to catalog
- ❌ Two-step process (provision, then add)
- ❌ Easy to forget to add to catalog

---

## 🎯 Proposed Architecture (Integrated)

```
rbee-hive handler
    ↓
WorkerCatalog::provision_and_add()
    ↓
Uses WorkerProvisioner internally
    ↓
Provisions worker
    ↓
Automatically adds to catalog
    ↓
Returns WorkerBinary (already cataloged)
```

**Benefits:**
- ✅ Single method call
- ✅ Automatic cataloging
- ✅ Cleaner API
- ✅ Follows model-catalog pattern

---

## 📝 Implementation

### 1. Add worker-provisioner Dependency

**File:** `worker-catalog/Cargo.toml`

```toml
[dependencies]
# ... existing deps ...
rbee-hive-worker-provisioner = { path = "../worker-provisioner" }
tokio-util = "0.7"  # For CancellationToken
```

### 2. Add provision_and_add() Method

**File:** `worker-catalog/src/lib.rs`

```rust
use rbee_hive_worker_provisioner::WorkerProvisioner;
use rbee_hive_artifact_catalog::ArtifactProvisioner;
use tokio_util::sync::CancellationToken;

impl WorkerCatalog {
    /// Provision a worker and automatically add it to the catalog
    ///
    /// This is a convenience method that:
    /// 1. Uses WorkerProvisioner to download/build the worker
    /// 2. Automatically adds the resulting WorkerBinary to the catalog
    /// 3. Returns the WorkerBinary
    ///
    /// # Arguments
    /// * `worker_id` - Worker ID (e.g., "llm-worker-rbee-cpu")
    /// * `job_id` - Job ID for narration routing
    /// * `cancel_token` - Cancellation token to abort provisioning
    ///
    /// # Example
    /// ```rust,no_run
    /// use rbee_hive_worker_catalog::WorkerCatalog;
    /// use tokio_util::sync::CancellationToken;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let catalog = WorkerCatalog::new()?;
    /// let cancel_token = CancellationToken::new();
    /// 
    /// let worker = catalog.provision_and_add(
    ///     "llm-worker-rbee-cpu",
    ///     "job-123",
    ///     cancel_token
    /// ).await?;
    /// 
    /// // Worker is already in catalog!
    /// assert!(catalog.contains(worker.id()));
    /// # Ok(())
    /// # }
    /// ```
    pub async fn provision_and_add(
        &self,
        worker_id: &str,
        job_id: &str,
        cancel_token: CancellationToken,
    ) -> Result<WorkerBinary> {
        // Create provisioner
        let provisioner = WorkerProvisioner::new()?;
        
        // Provision the worker
        let worker_binary = provisioner.provision(worker_id, job_id, cancel_token).await?;
        
        // Add to catalog
        self.add(worker_binary.clone())?;
        
        Ok(worker_binary)
    }
}
```

### 3. Optional: Add provision() Method (Without Adding)

For cases where you want to provision but not add to catalog:

```rust
impl WorkerCatalog {
    /// Provision a worker without adding it to the catalog
    ///
    /// Use this if you want to provision a worker but manage cataloging manually.
    /// Most users should use `provision_and_add()` instead.
    pub async fn provision(
        &self,
        worker_id: &str,
        job_id: &str,
        cancel_token: CancellationToken,
    ) -> Result<WorkerBinary> {
        let provisioner = WorkerProvisioner::new()?;
        provisioner.provision(worker_id, job_id, cancel_token).await
    }
}
```

---

## 🔄 Usage Comparison

### Before (Manual)

```rust
// rbee-hive handler
let provisioner = WorkerProvisioner::new()?;
let worker = provisioner.provision("llm-worker-rbee-cpu", "job-123", cancel_token).await?;

// Don't forget to add to catalog!
let catalog = WorkerCatalog::new()?;
catalog.add(worker)?;
```

### After (Integrated)

```rust
// rbee-hive handler
let catalog = WorkerCatalog::new()?;
let worker = catalog.provision_and_add("llm-worker-rbee-cpu", "job-123", cancel_token).await?;

// Worker is already cataloged!
```

---

## 🎯 Benefits

1. **Simpler API**
   - One method call instead of two
   - Less code in handlers

2. **Automatic Cataloging**
   - Can't forget to add to catalog
   - Consistent behavior

3. **Follows Patterns**
   - Similar to how model-catalog could work
   - Consistent with artifact-catalog abstraction

4. **Flexibility**
   - Still have access to WorkerProvisioner directly if needed
   - Can provision without cataloging if desired

---

## 📊 Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│ rbee-hive Handler                                        │
│                                                          │
│ catalog.provision_and_add(id, job_id, cancel_token)    │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ WorkerCatalog                                            │
│                                                          │
│ pub async fn provision_and_add(...) -> Result<...> {   │
│     let provisioner = WorkerProvisioner::new()?;        │
│     let worker = provisioner.provision(...).await?;     │
│     self.add(worker.clone())?;                          │
│     Ok(worker)                                           │
│ }                                                        │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ WorkerProvisioner                                        │
│                                                          │
│ - Fetch PKGBUILD from catalog                           │
│ - Parse PKGBUILD                                         │
│ - Fetch sources                                          │
│ - Build (if needed)                                      │
│ - Package                                                │
│ - Install binary                                         │
│ - Return WorkerBinary                                    │
└─────────────────────────────────────────────────────────┘
```

---

## ✅ Implementation Checklist

- [ ] Add `worker-provisioner` dependency to `worker-catalog/Cargo.toml`
- [ ] Add `tokio-util` dependency for `CancellationToken`
- [ ] Implement `provision_and_add()` method in `WorkerCatalog`
- [ ] Add tests for `provision_and_add()`
- [ ] Update documentation
- [ ] Update rbee-hive handlers to use new method

---

## 🚀 Next Steps

1. **Implement in worker-catalog**
   - Add dependencies
   - Add `provision_and_add()` method
   - Add tests

2. **Update rbee-hive**
   - Use `WorkerCatalog::provision_and_add()` instead of manual provisioning
   - Remove old worker_install.rs code
   - Update handlers

3. **Test**
   - Unit tests for `provision_and_add()`
   - Integration tests with rbee-hive
   - Test cancellation

---

**TEAM-402 - Provisioner Integration Design**

This is the right architecture! The catalog should own the provisioning process.
