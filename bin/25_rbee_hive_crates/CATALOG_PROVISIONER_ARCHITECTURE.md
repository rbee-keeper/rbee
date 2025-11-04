# Catalog + Provisioner Architecture

**Date:** 2025-11-04  
**Team:** TEAM-402  
**Status:** ✅ VERIFIED

---

## 🎯 The Correct Architecture

After discovering a circular dependency issue, we've confirmed the correct architecture:

### ✅ Catalog and Provisioner are SIBLINGS (Not Parent-Child)

```
rbee-hive
    ↓
Uses BOTH:
├── ModelCatalog (storage)
└── ModelProvisioner (downloading)
```

**NOT:**
```
❌ ModelCatalog → ModelProvisioner (circular dependency!)
```

---

## 🔄 Why Circular Dependency Occurs

1. **ModelProvisioner** needs `ModelEntry` type
2. **ModelEntry** is defined in `model-catalog`
3. **ModelProvisioner** depends on `model-catalog`
4. If **ModelCatalog** depends on `model-provisioner` → **CIRCULAR!**

---

## ✅ Solution: Keep Them Separate

### Dependencies

```
model-catalog
    ↓
artifact-catalog

model-provisioner
    ↓
├── artifact-catalog
└── model-catalog (for ModelEntry type)

rbee-hive
    ↓
├── model-catalog
└── model-provisioner
```

### Usage Pattern in rbee-hive

```rust
// rbee-hive handler
async fn handle_model_download(model_id: &str) -> Result<()> {
    // 1. Create provisioner
    let provisioner = ModelProvisioner::new()?;
    
    // 2. Provision the model
    let cancel_token = CancellationToken::new();
    let model = provisioner.provision(model_id, "job-123", cancel_token).await?;
    
    // 3. Add to catalog
    let catalog = ModelCatalog::new()?;
    catalog.add(model)?;
    
    Ok(())
}
```

---

## 📊 Test Status

### Model Catalog
- ✅ 1 test passing
- Tests CRUD operations

### Model Provisioner  
- ✅ 33 tests passing
- Tests HuggingFace vendor
- Tests provisioner logic
- Tests path handling

### Total
- ✅ **34 tests passing**
- ✅ No circular dependencies
- ✅ Clean architecture

---

## 🔧 Same Pattern for Workers

The **worker-catalog** and **worker-provisioner** should follow the same pattern:

```
worker-catalog
    ↓
artifact-catalog

worker-provisioner
    ↓
├── artifact-catalog
└── worker-catalog (for WorkerBinary type)

rbee-hive
    ↓
├── worker-catalog
└── worker-provisioner
```

---

## ✅ Benefits of This Architecture

1. **No Circular Dependencies**
   - Catalog doesn't depend on provisioner
   - Provisioner can depend on catalog (for types)

2. **Clear Separation**
   - Catalog = storage and retrieval
   - Provisioner = downloading/building

3. **Flexible**
   - Can use catalog without provisioner
   - Can use provisioner without catalog (for testing)
   - rbee-hive orchestrates both

4. **Testable**
   - Each component tests independently
   - No complex mocking needed

---

## 📝 Key Takeaway

**Catalog and Provisioner are SIBLINGS, not parent-child.**

rbee-hive is responsible for:
1. Calling provisioner to download/build
2. Calling catalog to store
3. Orchestrating the workflow

This keeps dependencies clean and components focused.

---

**TEAM-402 - Architecture Verified!** ✅
