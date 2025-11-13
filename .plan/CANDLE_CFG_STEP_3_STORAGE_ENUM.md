# Step 3: Add CFG Gates to Storage Enum

**Estimated Time:** 2 hours  
**Difficulty:** High (35 methods to update)  
**Dependencies:** Step 1, Step 2

---

## 🎯 OBJECTIVE

Add `#[cfg(feature = "...")]` gates to `Storage` enum and all 35 storage methods.

---

## 📝 FILE TO MODIFY

`candle-core/src/storage.rs` (~915 lines)

---

## 🔧 CHANGES REQUIRED

### 1. Storage Enum (lines 12-18)

**Before:**
```rust
#[derive(Debug)]
pub enum Storage {
    Cpu(CpuStorage),
    Cuda(CudaStorage),
    Metal(MetalStorage),
    #[cfg(feature = "rocm")]
    Rocm(RocmStorage),
}
```

**After:**
```rust
#[derive(Debug)]
pub enum Storage {
    #[cfg(feature = "cpu")]
    Cpu(CpuStorage),
    #[cfg(feature = "cuda")]
    Cuda(CudaStorage),
    #[cfg(feature = "metal")]
    Metal(MetalStorage),
    #[cfg(feature = "rocm")]
    Rocm(RocmStorage),
}
```

---

### 2. Import Statement (lines 4-6)

**Before:**
```rust
use crate::{CpuStorage, CudaStorage, DType, Device, Error, Layout, MetalStorage, Result, Shape};
#[cfg(feature = "rocm")]
use crate::RocmStorage;
```

**After:**
```rust
use crate::{DType, Device, Error, Layout, Result, Shape};
#[cfg(feature = "cpu")]
use crate::CpuStorage;
#[cfg(feature = "cuda")]
use crate::CudaStorage;
#[cfg(feature = "metal")]
use crate::MetalStorage;
#[cfg(feature = "rocm")]
use crate::RocmStorage;
```

---

### 3. Pattern for All 35 Methods

**General Pattern:**
```rust
pub(crate) fn method_name(&self, ...) -> Result<Self> {
    match self {
        #[cfg(feature = "cpu")]
        Storage::Cpu(storage) => {
            let storage = storage.method_name(...)?;
            Ok(Self::Cpu(storage))
        }
        #[cfg(feature = "cuda")]
        Self::Cuda(storage) => {
            let storage = storage.method_name(...)?;
            Ok(Self::Cuda(storage))
        }
        #[cfg(feature = "metal")]
        Self::Metal(storage) => {
            let storage = storage.method_name(...)?;
            Ok(Self::Metal(storage))
        }
        #[cfg(feature = "rocm")]
        Self::Rocm(storage) => {
            let storage = storage.method_name(...)?;
            Ok(Self::Rocm(storage))
        }
    }
}
```

---

### 4. List of 35 Methods to Update

#### Basic Methods (4)
1. ✅ `try_clone()` - lines 21-38
2. ✅ `device()` - lines 40-48
3. ✅ `dtype()` - lines 50-58
4. ✅ `const_set()` - lines 90-98

#### Math Operations (4)
5. ✅ `affine()` - lines 100-120
6. ✅ `powf()` - lines 122-142
7. ✅ `elu()` - lines 144-164
8. ✅ `cmp()` - lines 166-204

#### Core Operations (3)
9. ✅ `reduce_op()` - lines 206-226
10. ✅ `to_dtype()` - lines 228-248
11. ✅ `unary_impl()` - lines 388-408

#### Binary Operations (2)
12. ✅ `binary_impl()` - lines 410-446
13. ❌ `where_cond()` - lines 666-697 (3-way match)

#### Custom Ops (6)
14. ✅ `apply_op1()` - lines 250-270
15. ✅ `apply_op2()` - lines 272-300
16. ✅ `apply_op3()` - lines 302-333
17. ✅ `inplace_op1()` - lines 335-343
18. ✅ `inplace_op2()` - lines 345-361
19. ✅ `inplace_op3()` - lines 363-386

#### Convolution (4)
20. ✅ `conv1d()` - lines 448-483
21. ✅ `conv_transpose1d()` - lines 485-519
22. ✅ `conv2d()` - lines 521-555
23. ❌ `conv_transpose2d()` - lines 557-586

#### Pooling (2)
24. ❌ `avg_pool2d()` - lines 588-608
25. ❌ `max_pool2d()` - lines 610-630

#### Upsampling (2)
26. ❌ `upsample_nearest1d()` - lines 632-647
27. ❌ `upsample_nearest2d()` - lines 649-664

#### Indexing (5)
28. ❌ `gather()` - lines 699+
29. ❌ `scatter_set()` - lines ~730+
30. ❌ `scatter_add()` - lines ~760+
31. ❌ `index_add()` - lines ~790+
32. ❌ `index_select()` - lines ~820+

#### Matrix Operations (1)
33. ❌ `matmul()` - lines ~850+

#### Memory Operations (2)
34. ❌ `copy_strided_src()` - lines ~880+
35. ❌ `copy2d()` - lines ~900+

**Legend:**
- ✅ Already has ROCm branch (from TEAM-501)
- ❌ Needs cfg gates added

---

### 5. Special Cases

#### `where_cond()` - 3-way match

**Pattern:**
```rust
pub(crate) fn where_cond(...) -> Result<Self> {
    self.same_device(t, "where")?;
    self.same_device(f, "where")?;
    t.same_dtype(f, "where")?;
    match (self, t, f) {
        #[cfg(feature = "cpu")]
        (Storage::Cpu(cond), Storage::Cpu(t), Storage::Cpu(f)) => {
            let storage = cond.where_cond(layout, t, layout_t, f, layout_f)?;
            Ok(Self::Cpu(storage))
        }
        #[cfg(feature = "cuda")]
        (Self::Cuda(cond), Self::Cuda(t), Self::Cuda(f)) => {
            let storage = cond.where_cond(layout, t, layout_t, f, layout_f)?;
            Ok(Self::Cuda(storage))
        }
        #[cfg(feature = "metal")]
        (Self::Metal(cond), Self::Metal(t), Self::Metal(f)) => {
            let storage = cond.where_cond(layout, t, layout_t, f, layout_f)?;
            Ok(Self::Metal(storage))
        }
        #[cfg(feature = "rocm")]
        (Self::Rocm(cond), Self::Rocm(t), Self::Rocm(f)) => {
            let storage = cond.where_cond(layout, t, layout_t, f, layout_f)?;
            Ok(Self::Rocm(storage))
        }
        (_, lhs, rhs) => Err(Error::DeviceMismatchBinaryOp {
            lhs: lhs.device().location(),
            rhs: rhs.device().location(),
            op: "where",
        }
        .bt()),
    }
}
```

---

## 🤖 AUTOMATION SCRIPT

Since this is repetitive, here's a script to help:

```bash
#!/bin/bash
# add_cfg_gates_to_storage.sh

FILE="candle-core/src/storage.rs"

# Add cfg gates to match arms
sed -i 's/Storage::Cpu(/#[cfg(feature = "cpu")]\n        Storage::Cpu(/g' "$FILE"
sed -i 's/Self::Cuda(/#[cfg(feature = "cuda")]\n        Self::Cuda(/g' "$FILE"
sed -i 's/Self::Metal(/#[cfg(feature = "metal")]\n        Self::Metal(/g' "$FILE"

echo "Done! Review changes with: git diff $FILE"
```

**⚠️ WARNING:** This script is a starting point. **MANUAL REVIEW REQUIRED!**

---

## ✅ VERIFICATION

```bash
# CPU-only build
cargo check --no-default-features --features cpu

# CUDA-only build
cargo check --no-default-features --features cuda

# Metal-only build
cargo check --no-default-features --features metal

# ROCm-only build
cargo check --no-default-features --features rocm

# All backends
cargo check --features all-backends
```

---

## 🚨 COMMON ISSUES

### Issue 1: Unreachable pattern in 3-way matches
```
error: unreachable pattern
```
**Fix:** Make sure the wildcard pattern `(_, lhs, rhs)` is at the end

### Issue 2: Missing cfg gate on error branches
```
error: variant `Cpu` not found
```
**Fix:** Error branches in binary ops need cfg gates too

---

## 📊 PROGRESS TRACKING

- [ ] Update import statements
- [ ] Add cfg gates to `Storage` enum
- [ ] Update 21 methods already done by TEAM-501
- [ ] Update `conv_transpose2d()`
- [ ] Update `avg_pool2d()`
- [ ] Update `max_pool2d()`
- [ ] Update `upsample_nearest1d()`
- [ ] Update `upsample_nearest2d()`
- [ ] Update `where_cond()`
- [ ] Update `gather()`
- [ ] Update `scatter_set()`
- [ ] Update `scatter_add()`
- [ ] Update `index_add()`
- [ ] Update `index_select()`
- [ ] Update `matmul()`
- [ ] Update `copy_strided_src()`
- [ ] Update `copy2d()`
- [ ] Run verification commands
- [ ] Commit changes

---

## 🎯 NEXT STEP

**Proceed to STEP_4_CUSTOM_OPS.md**

---

**TEAM-501 STEP 3**
