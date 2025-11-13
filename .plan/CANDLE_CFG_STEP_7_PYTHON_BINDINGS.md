# Step 7: Update Python Bindings

**Estimated Time:** 1 hour  
**Difficulty:** Medium  
**Dependencies:** Steps 1-6

---

## 🎯 OBJECTIVE

Update `candle-pyo3` to support feature-gated backends.

---

## 📝 FILES TO MODIFY

1. `candle-pyo3/Cargo.toml`
2. `candle-pyo3/src/lib.rs`

---

## 🔧 CHANGES REQUIRED

### 1. Cargo.toml Features

**File:** `candle-pyo3/Cargo.toml`

**Before:**
```toml
[features]
default = []
cuda = ["candle-core/cuda"]
cudnn = ["candle-core/cudnn"]
mkl = ["candle-core/mkl"]
accelerate = ["candle-core/accelerate"]
metal = ["candle-core/metal"]
```

**After:**
```toml
[features]
default = ["cpu"]

# Backend features
cpu = ["candle-core/cpu"]
cuda = ["candle-core/cuda"]
cudnn = ["candle-core/cudnn"]
metal = ["candle-core/metal"]
rocm = ["candle-core/rocm"]

# Accelerator features
mkl = ["candle-core/mkl"]
accelerate = ["candle-core/accelerate"]

# Convenience features
all-backends = ["cpu", "cuda", "metal", "rocm"]
gpu-backends = ["cuda", "metal", "rocm"]
```

---

### 2. Python Device Class

**File:** `candle-pyo3/src/lib.rs`

**Before:**
```rust
#[pyclass]
pub struct PyDevice {
    device: Device,
}

#[pymethods]
impl PyDevice {
    #[staticmethod]
    fn cpu() -> Self {
        Self { device: Device::Cpu }
    }
    
    #[staticmethod]
    fn cuda(ordinal: usize) -> PyResult<Self> {
        Ok(Self { device: Device::new_cuda(ordinal)? })
    }
    
    #[staticmethod]
    fn metal(ordinal: usize) -> PyResult<Self> {
        Ok(Self { device: Device::new_metal(ordinal)? })
    }
}
```

**After:**
```rust
#[pyclass]
pub struct PyDevice {
    device: Device,
}

#[pymethods]
impl PyDevice {
    #[cfg(feature = "cpu")]
    #[staticmethod]
    fn cpu() -> Self {
        Self { device: Device::Cpu }
    }
    
    #[cfg(feature = "cuda")]
    #[staticmethod]
    fn cuda(ordinal: usize) -> PyResult<Self> {
        Ok(Self { device: Device::new_cuda(ordinal)? })
    }
    
    #[cfg(feature = "metal")]
    #[staticmethod]
    fn metal(ordinal: usize) -> PyResult<Self> {
        Ok(Self { device: Device::new_metal(ordinal)? })
    }
    
    #[cfg(feature = "rocm")]
    #[staticmethod]
    fn rocm(ordinal: usize) -> PyResult<Self> {
        Ok(Self { device: Device::new_rocm(ordinal)? })
    }
    
    #[cfg(feature = "cuda")]
    fn is_cuda(&self) -> bool {
        self.device.is_cuda()
    }
    
    #[cfg(feature = "metal")]
    fn is_metal(&self) -> bool {
        self.device.is_metal()
    }
    
    #[cfg(feature = "rocm")]
    fn is_rocm(&self) -> bool {
        self.device.is_rocm()
    }
    
    #[cfg(feature = "cpu")]
    fn is_cpu(&self) -> bool {
        matches!(self.device, Device::Cpu)
    }
}
```

---

### 3. Python Module Registration

**Before:**
```rust
#[pymodule]
fn candle(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyDevice>()?;
    m.add_class::<PyTensor>()?;
    // ... other classes
    Ok(())
}
```

**After:**
```rust
#[pymodule]
fn candle(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyDevice>()?;
    m.add_class::<PyTensor>()?;
    
    // Add backend availability flags
    #[cfg(feature = "cpu")]
    m.add("HAS_CPU", true)?;
    #[cfg(not(feature = "cpu"))]
    m.add("HAS_CPU", false)?;
    
    #[cfg(feature = "cuda")]
    m.add("HAS_CUDA", true)?;
    #[cfg(not(feature = "cuda"))]
    m.add("HAS_CUDA", false)?;
    
    #[cfg(feature = "metal")]
    m.add("HAS_METAL", true)?;
    #[cfg(not(feature = "metal"))]
    m.add("HAS_METAL", false)?;
    
    #[cfg(feature = "rocm")]
    m.add("HAS_ROCM", true)?;
    #[cfg(not(feature = "rocm"))]
    m.add("HAS_ROCM", false)?;
    
    Ok(())
}
```

---

### 4. Python Usage Example

**Python code:**
```python
import candle

# Check backend availability
print(f"CPU support: {candle.HAS_CPU}")
print(f"CUDA support: {candle.HAS_CUDA}")
print(f"Metal support: {candle.HAS_METAL}")
print(f"ROCm support: {candle.HAS_ROCM}")

# Create device (will fail if backend not compiled)
if candle.HAS_CUDA:
    device = candle.Device.cuda(0)
elif candle.HAS_METAL:
    device = candle.Device.metal(0)
elif candle.HAS_ROCM:
    device = candle.Device.rocm(0)
else:
    device = candle.Device.cpu()

# Create tensor
tensor = candle.Tensor([1.0, 2.0, 3.0], device=device)
```

---

## ✅ VERIFICATION

```bash
# Build Python wheel with CPU only
cd candle-pyo3
maturin build --release --no-default-features --features cpu

# Build Python wheel with CUDA
maturin build --release --features cuda

# Test in Python
python3 -c "import candle; print(candle.HAS_CUDA)"
```

---

## 📊 PROGRESS TRACKING

- [ ] Update `Cargo.toml` features
- [ ] Add cfg gates to `PyDevice` methods
- [ ] Add backend availability flags
- [ ] Update Python module registration
- [ ] Build Python wheels
- [ ] Test in Python
- [ ] Commit changes

---

## 🎯 NEXT STEP

**Proceed to STEP_8_EXAMPLES.md**

---

**TEAM-501 STEP 7**
