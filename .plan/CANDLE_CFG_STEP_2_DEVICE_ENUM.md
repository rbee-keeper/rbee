# Step 2: Add CFG Gates to Device Enum

**Estimated Time:** 1 hour  
**Difficulty:** Medium  
**Dependencies:** Step 1 (Feature Definitions)

---

## 🎯 OBJECTIVE

Add `#[cfg(feature = "...")]` gates to `Device` and `DeviceLocation` enums and all related methods.

---

## 📝 FILE TO MODIFY

`candle-core/src/device.rs` (~543 lines)

---

## 🔧 CHANGES REQUIRED

### 1. DeviceLocation Enum (lines 8-13)

**Before:**
```rust
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DeviceLocation {
    Cpu,
    Cuda { gpu_id: usize },
    Metal { gpu_id: usize },
    Rocm { gpu_id: usize },
}
```

**After:**
```rust
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DeviceLocation {
    #[cfg(feature = "cpu")]
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda { gpu_id: usize },
    #[cfg(feature = "metal")]
    Metal { gpu_id: usize },
    #[cfg(feature = "rocm")]
    Rocm { gpu_id: usize },
}
```

---

### 2. Device Enum (lines 17-23)

**Before:**
```rust
#[derive(Debug, Clone)]
pub enum Device {
    Cpu,
    Cuda(crate::CudaDevice),
    Metal(crate::MetalDevice),
    #[cfg(feature = "rocm")]
    Rocm(crate::RocmDevice),
}
```

**After:**
```rust
#[derive(Debug, Clone)]
pub enum Device {
    #[cfg(feature = "cpu")]
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(crate::CudaDevice),
    #[cfg(feature = "metal")]
    Metal(crate::MetalDevice),
    #[cfg(feature = "rocm")]
    Rocm(crate::RocmDevice),
}
```

---

### 3. Device Methods (9 methods to update)

#### Method 1: `as_cuda_device()` (lines ~242-249)

**Before:**
```rust
pub fn as_cuda_device(&self) -> Result<&crate::CudaDevice> {
    match self {
        Self::Cuda(d) => Ok(d),
        Self::Cpu => crate::bail!("expected a cuda device, got cpu"),
        Self::Metal(_) => crate::bail!("expected a cuda device, got Metal"),
        #[cfg(feature = "rocm")]
        Self::Rocm(_) => crate::bail!("expected a cuda device, got ROCm"),
    }
}
```

**After:**
```rust
#[cfg(feature = "cuda")]
pub fn as_cuda_device(&self) -> Result<&crate::CudaDevice> {
    match self {
        Self::Cuda(d) => Ok(d),
        #[cfg(feature = "cpu")]
        Self::Cpu => crate::bail!("expected a cuda device, got cpu"),
        #[cfg(feature = "metal")]
        Self::Metal(_) => crate::bail!("expected a cuda device, got Metal"),
        #[cfg(feature = "rocm")]
        Self::Rocm(_) => crate::bail!("expected a cuda device, got ROCm"),
    }
}
```

#### Method 2: `as_metal_device()` (lines ~251-259)

**Before:**
```rust
pub fn as_metal_device(&self) -> Result<&crate::MetalDevice> {
    match self {
        Self::Cuda(_) => crate::bail!("expected a metal device, got cuda"),
        Self::Cpu => crate::bail!("expected a metal device, got cpu"),
        Self::Metal(d) => Ok(d),
        #[cfg(feature = "rocm")]
        Self::Rocm(_) => crate::bail!("expected a metal device, got ROCm"),
    }
}
```

**After:**
```rust
#[cfg(feature = "metal")]
pub fn as_metal_device(&self) -> Result<&crate::MetalDevice> {
    match self {
        #[cfg(feature = "cuda")]
        Self::Cuda(_) => crate::bail!("expected a metal device, got cuda"),
        #[cfg(feature = "cpu")]
        Self::Cpu => crate::bail!("expected a metal device, got cpu"),
        Self::Metal(d) => Ok(d),
        #[cfg(feature = "rocm")]
        Self::Rocm(_) => crate::bail!("expected a metal device, got ROCm"),
    }
}
```

#### Method 3: `as_rocm_device()` (already has cfg gate)

**No changes needed** - already gated with `#[cfg(feature = "rocm")]`

#### Method 4: `new_cuda()` (needs cfg gate)

**Add:**
```rust
#[cfg(feature = "cuda")]
pub fn new_cuda(ordinal: usize) -> Result<Self> {
    Ok(Self::Cuda(crate::CudaDevice::new(ordinal)?))
}
```

#### Method 5: `new_metal()` (needs cfg gate)

**Add:**
```rust
#[cfg(feature = "metal")]
pub fn new_metal(ordinal: usize) -> Result<Self> {
    Ok(Self::Metal(crate::MetalDevice::new(ordinal)?))
}
```

#### Method 6: `new_rocm()` (already has cfg gate)

**No changes needed**

#### Method 7: `set_seed()` (lines ~287-294)

**Before:**
```rust
pub fn set_seed(&self, seed: u64) -> Result<()> {
    match self {
        Self::Cpu => CpuDevice.set_seed(seed),
        Self::Cuda(c) => c.set_seed(seed),
        Self::Metal(m) => m.set_seed(seed),
        #[cfg(feature = "rocm")]
        Self::Rocm(_) => Ok(()),
    }
}
```

**After:**
```rust
pub fn set_seed(&self, seed: u64) -> Result<()> {
    match self {
        #[cfg(feature = "cpu")]
        Self::Cpu => CpuDevice.set_seed(seed),
        #[cfg(feature = "cuda")]
        Self::Cuda(c) => c.set_seed(seed),
        #[cfg(feature = "metal")]
        Self::Metal(m) => m.set_seed(seed),
        #[cfg(feature = "rocm")]
        Self::Rocm(_) => Ok(()),
    }
}
```

#### Method 8: `same_device()` (lines ~297-305)

**After:**
```rust
pub fn same_device(&self, rhs: &Self) -> bool {
    match (self, rhs) {
        #[cfg(feature = "cpu")]
        (Self::Cpu, Self::Cpu) => true,
        #[cfg(feature = "cuda")]
        (Self::Cuda(lhs), Self::Cuda(rhs)) => lhs.same_device(rhs),
        #[cfg(feature = "metal")]
        (Self::Metal(lhs), Self::Metal(rhs)) => lhs.same_device(rhs),
        #[cfg(feature = "rocm")]
        (Self::Rocm(lhs), Self::Rocm(rhs)) => lhs == rhs,
        _ => false,
    }
}
```

#### Method 9: `location()` (lines ~308-315)

**After:**
```rust
pub fn location(&self) -> DeviceLocation {
    match self {
        #[cfg(feature = "cpu")]
        Self::Cpu => DeviceLocation::Cpu,
        #[cfg(feature = "cuda")]
        Self::Cuda(device) => device.location(),
        #[cfg(feature = "metal")]
        Device::Metal(device) => device.location(),
        #[cfg(feature = "rocm")]
        Device::Rocm(device) => DeviceLocation::Rocm { gpu_id: device.id() },
    }
}
```

---

### 4. Helper Methods

#### `is_cpu()` (needs cfg gate)

**Add:**
```rust
#[cfg(feature = "cpu")]
pub fn is_cpu(&self) -> bool {
    matches!(self, Self::Cpu)
}
```

#### `is_cuda()` (needs cfg gate)

**Add:**
```rust
#[cfg(feature = "cuda")]
pub fn is_cuda(&self) -> bool {
    matches!(self, Self::Cuda(_))
}
```

#### `is_metal()` (needs cfg gate)

**Add:**
```rust
#[cfg(feature = "metal")]
pub fn is_metal(&self) -> bool {
    matches!(self, Self::Metal(_))
}
```

#### `is_rocm()` (already has cfg gate)

**No changes needed**

---

## ✅ VERIFICATION

```bash
# CPU-only build (should work)
cargo check --no-default-features --features cpu

# CUDA-only build (should work)
cargo check --no-default-features --features cuda

# Metal-only build (should work)
cargo check --no-default-features --features metal

# ROCm-only build (should work)
cargo check --no-default-features --features rocm

# Multi-backend build
cargo check --features "cpu,cuda"
```

---

## 🚨 COMMON ISSUES

### Issue 1: Unreachable pattern
```
error: unreachable pattern
```
**Fix:** Make sure all match arms have cfg gates

### Issue 2: Missing variant
```
error: variant `Cpu` not found
```
**Fix:** Add `#[cfg(feature = "cpu")]` to the match arm

---

## 📊 PROGRESS TRACKING

- [ ] Add cfg gates to `DeviceLocation` enum
- [ ] Add cfg gates to `Device` enum
- [ ] Update `as_cuda_device()` method
- [ ] Update `as_metal_device()` method
- [ ] Update `new_cuda()` method
- [ ] Update `new_metal()` method
- [ ] Update `set_seed()` method
- [ ] Update `same_device()` method
- [ ] Update `location()` method
- [ ] Update `is_cpu()` method
- [ ] Update `is_cuda()` method
- [ ] Update `is_metal()` method
- [ ] Run verification commands
- [ ] Commit changes

---

## 🎯 NEXT STEP

**Proceed to STEP_3_STORAGE_ENUM.md**

---

**TEAM-501 STEP 2**
