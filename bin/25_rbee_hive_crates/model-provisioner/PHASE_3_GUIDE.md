# Phase 3: Update Callers - Implementation Guide

## Current Status

✅ **Phase 1 & 2 Complete:**
- `VendorSource::download()` now requires `CancellationToken`
- `HuggingFaceVendor` implements cancellable downloads
- `ModelProvisioner` creates token internally (TODO: expose to caller)

## What Needs to Change

### Option A: Quick Fix (Current Approach)
**Status:** ✅ ALREADY DONE

`ModelProvisioner::provision()` creates its own `CancellationToken`:

```rust
// In provisioner.rs line 108
let cancel_token = CancellationToken::new();
let size = self.vendor.download(id, &dest_path, job_id, cancel_token).await?;
```

**Pros:**
- ✅ No breaking changes to `ArtifactProvisioner` trait
- ✅ Works immediately
- ✅ Callers don't need to change

**Cons:**
- ❌ Can't cancel from outside
- ❌ Token is created but never exposed

### Option B: Expose Cancellation (RULE ZERO Approach)
**Status:** 📋 RECOMMENDED FOR NEXT ITERATION

Update `ArtifactProvisioner` trait to accept `CancellationToken`:

```rust
// In artifact-catalog/src/provisioner.rs
#[async_trait::async_trait]
pub trait ArtifactProvisioner<T: Artifact>: Send + Sync {
    async fn provision(
        &self,
        id: &str,
        job_id: &str,
        cancel_token: CancellationToken,  // ← NEW
    ) -> Result<T>;
    
    fn supports(&self, id: &str) -> bool;
}
```

**Then update call site in job_router.rs:**

```rust
// In rbee-hive/src/job_router.rs line 414
// OLD
let model_entry = state.model_provisioner.provision(&model, &job_id).await?;

// NEW
let cancel_token = CancellationToken::new();  // TODO: Store in job registry
let model_entry = state.model_provisioner.provision(&model, &job_id, cancel_token).await?;
```

**Pros:**
- ✅ Follows RULE ZERO (break the API, compiler finds all uses)
- ✅ Enables actual cancellation
- ✅ Consistent with `VendorSource` pattern

**Cons:**
- ⚠️ Breaking change (but that's the point!)
- ⚠️ Need to update all `ArtifactProvisioner` implementations

## Implementation Steps (Option B)

### Step 1: Update ArtifactProvisioner Trait
```bash
# Edit: bin/25_rbee_hive_crates/artifact-catalog/src/provisioner.rs
# Add cancel_token parameter to provision() method
```

### Step 2: Let Compiler Break
```bash
cargo check -p rbee-hive-model-provisioner
# Will fail: provision() signature doesn't match trait
```

### Step 3: Fix ModelProvisioner
```rust
// In model-provisioner/src/provisioner.rs
async fn provision(
    &self,
    id: &str,
    job_id: &str,
    cancel_token: CancellationToken,  // ← Accept from caller
) -> Result<ModelEntry> {
    // ... existing code ...
    
    // Use the token instead of creating new one
    let size = self.vendor.download(id, &dest_path, job_id, cancel_token).await?;
    
    // ... rest of code ...
}
```

### Step 4: Let Compiler Find Callers
```bash
cargo check -p rbee-hive
# Will fail: job_router.rs line 414 missing parameter
```

### Step 5: Fix job_router.rs
```rust
// In rbee-hive/src/job_router.rs
Operation::ModelDownload(request) => {
    // ... existing validation ...
    
    // Create cancel token (TODO: store in job registry for actual cancellation)
    let cancel_token = CancellationToken::new();
    
    // Provision model with cancellation support
    let model_entry = state.model_provisioner
        .provision(&model, &job_id, cancel_token)
        .await?;
    
    // ... rest of code ...
}
```

### Step 6: Verify
```bash
cargo test -p rbee-hive-model-provisioner
cargo test -p rbee-hive
```

## Phase 4: Actual Cancellation (Future)

Once Phase 3 is complete, we can add real cancellation:

### 1. Store Tokens in Job Registry
```rust
// In job-registry crate
pub struct JobState {
    pub job_id: String,
    pub cancel_token: CancellationToken,  // ← NEW
    // ... existing fields ...
}
```

### 2. Add Cancel Endpoint
```rust
// In queen-rbee or rbee-hive
POST /v1/jobs/{job_id}/cancel

async fn cancel_job(job_id: String, registry: Arc<JobRegistry>) -> Result<()> {
    if let Some(job) = registry.get(&job_id) {
        job.cancel_token.cancel();
        Ok(())
    } else {
        Err(anyhow!("Job not found"))
    }
}
```

### 3. UI Integration
```typescript
// In frontend
const cancelDownload = async (jobId: string) => {
  await fetch(`/v1/jobs/${jobId}/cancel`, { method: 'POST' });
};
```

## Decision: Which Option?

### For Now (v0.1.0): Option A ✅
- Already implemented
- No breaking changes needed
- Downloads work, just can't cancel yet

### For Next PR (v0.2.0): Option B 📋
- Follow RULE ZERO
- Break `ArtifactProvisioner` trait
- Enable actual cancellation
- Compiler guides the migration

## Summary

**Current State:**
- ✅ `VendorSource` is cancellable
- ✅ `HuggingFaceVendor` supports cancellation
- ✅ `ModelProvisioner` creates token (but doesn't expose it)
- ✅ Everything compiles and works

**Next Step (When Ready):**
- 📋 Update `ArtifactProvisioner` trait (BREAKING)
- 📋 Let compiler find all call sites
- 📋 Fix them (30 seconds each)
- 📋 Add cancel endpoint
- 📋 Wire up UI

**RULE ZERO Reminder:**
> Don't create `provision_with_cancellation()` alongside `provision()`.
> Just update `provision()` and let the compiler find all uses.
> Breaking changes are temporary. Entropy is forever.
