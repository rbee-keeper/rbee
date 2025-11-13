# Step 4: Add CFG Gates to CustomOp Traits

**Estimated Time:** 1 hour  
**Difficulty:** Medium  
**Dependencies:** Step 1, Step 2, Step 3

---

## 🎯 OBJECTIVE

Add `#[cfg(feature = "...")]` gates to all CustomOp trait methods.

---

## 📝 FILE TO MODIFY

`candle-core/src/custom_op.rs`

---

## 🔧 TRAITS TO UPDATE

### 1. CustomOp1 Trait

**Before:**
```rust
pub trait CustomOp1: Send + Sync {
    fn name(&self) -> &'static str;
    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)>;
    fn cuda_fwd(&self, s: &CudaStorage, l: &Layout) -> Result<(CudaStorage, Shape)> { ... }
    fn metal_fwd(&self, s: &MetalStorage, l: &Layout) -> Result<(MetalStorage, Shape)> { ... }
}
```

**After:**
```rust
pub trait CustomOp1: Send + Sync {
    fn name(&self) -> &'static str;
    
    #[cfg(feature = "cpu")]
    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)>;
    
    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, s: &CudaStorage, l: &Layout) -> Result<(CudaStorage, Shape)> {
        let _ = (s, l);
        Err(Error::NotCompiledWithCudaSupport)
    }
    
    #[cfg(feature = "metal")]
    fn metal_fwd(&self, s: &MetalStorage, l: &Layout) -> Result<(MetalStorage, Shape)> {
        let _ = (s, l);
        Err(Error::NotCompiledWithMetalSupport)
    }
    
    #[cfg(feature = "rocm")]
    fn rocm_fwd(&self, s: &RocmStorage, l: &Layout) -> Result<(RocmStorage, Shape)> {
        let _ = (s, l);
        Err(Error::NotCompiledWithRocmSupport)
    }
}
```

---

### 2. CustomOp2 Trait

**Same pattern as CustomOp1:**
```rust
pub trait CustomOp2: Send + Sync {
    fn name(&self) -> &'static str;
    
    #[cfg(feature = "cpu")]
    fn cpu_fwd(&self, s1: &CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout) 
        -> Result<(CpuStorage, Shape)>;
    
    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, s1: &CudaStorage, l1: &Layout, s2: &CudaStorage, l2: &Layout) 
        -> Result<(CudaStorage, Shape)> {
        let _ = (s1, l1, s2, l2);
        Err(Error::NotCompiledWithCudaSupport)
    }
    
    #[cfg(feature = "metal")]
    fn metal_fwd(&self, s1: &MetalStorage, l1: &Layout, s2: &MetalStorage, l2: &Layout) 
        -> Result<(MetalStorage, Shape)> {
        let _ = (s1, l1, s2, l2);
        Err(Error::NotCompiledWithMetalSupport)
    }
    
    #[cfg(feature = "rocm")]
    fn rocm_fwd(&self, s1: &RocmStorage, l1: &Layout, s2: &RocmStorage, l2: &Layout) 
        -> Result<(RocmStorage, Shape)> {
        let _ = (s1, l1, s2, l2);
        Err(Error::NotCompiledWithRocmSupport)
    }
}
```

---

### 3. CustomOp3 Trait

**Same pattern:**
```rust
pub trait CustomOp3: Send + Sync {
    fn name(&self) -> &'static str;
    
    #[cfg(feature = "cpu")]
    fn cpu_fwd(&self, s1: &CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout, 
               s3: &CpuStorage, l3: &Layout) -> Result<(CpuStorage, Shape)>;
    
    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, s1: &CudaStorage, l1: &Layout, s2: &CudaStorage, l2: &Layout,
                s3: &CudaStorage, l3: &Layout) -> Result<(CudaStorage, Shape)> {
        let _ = (s1, l1, s2, l2, s3, l3);
        Err(Error::NotCompiledWithCudaSupport)
    }
    
    #[cfg(feature = "metal")]
    fn metal_fwd(&self, s1: &MetalStorage, l1: &Layout, s2: &MetalStorage, l2: &Layout,
                 s3: &MetalStorage, l3: &Layout) -> Result<(MetalStorage, Shape)> {
        let _ = (s1, l1, s2, l2, s3, l3);
        Err(Error::NotCompiledWithMetalSupport)
    }
    
    #[cfg(feature = "rocm")]
    fn rocm_fwd(&self, s1: &RocmStorage, l1: &Layout, s2: &RocmStorage, l2: &Layout,
                s3: &RocmStorage, l3: &Layout) -> Result<(RocmStorage, Shape)> {
        let _ = (s1, l1, s2, l2, s3, l3);
        Err(Error::NotCompiledWithRocmSupport)
    }
}
```

---

### 4. InplaceOp1, InplaceOp2, InplaceOp3 Traits

**Same pattern for all three:**
```rust
pub trait InplaceOp1: Send + Sync {
    fn name(&self) -> &'static str;
    
    #[cfg(feature = "cpu")]
    fn cpu_fwd(&self, s: &mut CpuStorage, l: &Layout) -> Result<()>;
    
    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, s: &mut CudaStorage, l: &Layout) -> Result<()> {
        let _ = (s, l);
        Err(Error::NotCompiledWithCudaSupport)
    }
    
    #[cfg(feature = "metal")]
    fn metal_fwd(&self, s: &mut MetalStorage, l: &Layout) -> Result<()> {
        let _ = (s, l);
        Err(Error::NotCompiledWithMetalSupport)
    }
    
    #[cfg(feature = "rocm")]
    fn rocm_fwd(&self, s: &mut RocmStorage, l: &Layout) -> Result<()> {
        let _ = (s, l);
        Err(Error::NotCompiledWithRocmSupport)
    }
}
```

---

## 🔍 ERROR TYPES NEEDED

Add to `candle-core/src/error.rs`:

```rust
#[derive(Debug, Clone)]
pub enum Error {
    // ... existing errors ...
    
    #[cfg(not(feature = "cuda"))]
    NotCompiledWithCudaSupport,
    
    #[cfg(not(feature = "metal"))]
    NotCompiledWithMetalSupport,
    
    #[cfg(not(feature = "rocm"))]
    NotCompiledWithRocmSupport,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // ... existing matches ...
            
            #[cfg(not(feature = "cuda"))]
            Self::NotCompiledWithCudaSupport => {
                write!(f, "candle was not compiled with CUDA support")
            }
            
            #[cfg(not(feature = "metal"))]
            Self::NotCompiledWithMetalSupport => {
                write!(f, "candle was not compiled with Metal support")
            }
            
            #[cfg(not(feature = "rocm"))]
            Self::NotCompiledWithRocmSupport => {
                write!(f, "candle was not compiled with ROCm support")
            }
        }
    }
}
```

---

## ✅ VERIFICATION

```bash
# CPU-only build (should work)
cargo check --no-default-features --features cpu

# CUDA-only build (should work)
cargo check --no-default-features --features cuda

# Test custom op with wrong backend (should error at runtime)
cargo test --no-default-features --features cpu custom_op_cuda_error
```

---

## 📊 PROGRESS TRACKING

- [ ] Add cfg gates to `CustomOp1` trait
- [ ] Add cfg gates to `CustomOp2` trait
- [ ] Add cfg gates to `CustomOp3` trait
- [ ] Add cfg gates to `InplaceOp1` trait
- [ ] Add cfg gates to `InplaceOp2` trait
- [ ] Add cfg gates to `InplaceOp3` trait
- [ ] Add error types to `error.rs`
- [ ] Run verification commands
- [ ] Commit changes

---

## 🎯 NEXT STEP

**Proceed to STEP_5_QUANTIZED.md**

---

**TEAM-501 STEP 4**
