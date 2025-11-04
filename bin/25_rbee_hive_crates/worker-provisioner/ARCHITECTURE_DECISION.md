# Architecture Decision: Provisioner Integration

**Date:** 2025-11-04  
**Team:** TEAM-402  
**Decision:** WorkerCatalog should own WorkerProvisioner

---

## 🤔 The Question

**Where should WorkerProvisioner be used?**

### Option A: Standalone (What I Built)
```rust
// rbee-hive handler
let provisioner = WorkerProvisioner::new()?;
let worker = provisioner.provision(...).await?;
catalog.add(worker)?;  // Manual step
```

### Option B: Integrated into Catalog (Better!)
```rust
// rbee-hive handler
let catalog = WorkerCatalog::new()?;
let worker = catalog.provision_and_add(...).await?;
// Worker is already cataloged!
```

---

## ✅ Decision: Option B (Integrated)

**Rationale:**
1. **Simpler API** - One method call instead of two
2. **Automatic cataloging** - Can't forget to add to catalog
3. **Consistent with patterns** - Catalog owns its artifacts
4. **Better encapsulation** - Provisioner is an implementation detail

---

## 🏗️ Implementation Plan

### Phase 5a: Integrate Provisioner into Catalog

1. **Add dependency** to `worker-catalog/Cargo.toml`:
   ```toml
   rbee-hive-worker-provisioner = { path = "../worker-provisioner" }
   ```

2. **Add method** to `WorkerCatalog`:
   ```rust
   pub async fn provision_and_add(
       &self,
       worker_id: &str,
       job_id: &str,
       cancel_token: CancellationToken,
   ) -> Result<WorkerBinary>
   ```

3. **Update rbee-hive** to use `catalog.provision_and_add()`

### Phase 5b: Clean Up rbee-hive

1. Remove old files:
   - `src/pkgbuild_parser.rs`
   - `src/pkgbuild_executor.rs`
   - `src/source_fetcher.rs`
   - `src/worker_install.rs`

2. Update handlers to use new API

---

## 📊 Dependency Graph

```
rbee-hive
    ↓
worker-catalog (public API)
    ↓
worker-provisioner (internal implementation)
    ↓
artifact-catalog (shared abstraction)
```

**Key Point:** rbee-hive only depends on `worker-catalog`, not `worker-provisioner` directly!

---

## ✅ Benefits

1. **Clean separation of concerns**
   - Catalog = public API
   - Provisioner = internal implementation

2. **Easier to use**
   - One method call
   - Automatic cataloging

3. **Easier to test**
   - Mock the catalog
   - Don't need to mock provisioner

4. **Future-proof**
   - Can swap provisioner implementation
   - Can add caching, validation, etc.

---

**TEAM-402 - Architecture Decision Complete**

The provisioner should be integrated into the catalog, not used standalone!
