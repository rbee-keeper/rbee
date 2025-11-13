# Phase 2: Kernel Compilation System

**Duration:** Week 2-3 (10-14 days)  
**Team:** TEAM-488  
**Status:** 📋 READY TO START

---

## Goal

Translate CUDA kernels to HIP, compile to `.hsaco` binaries, and create a build system to embed them in Rust.

**Success Criteria:**
- ✅ All 11 CUDA kernels translated to HIP
- ✅ All kernels compile to `.hsaco` binaries
- ✅ Build system embeds binaries in Rust
- ✅ Runtime kernel loading works
- ✅ Basic kernel tests pass

---

## Strategy (Based on rocm-rs)

**How rocm-rs does it:**
1. Write kernel in C++ (or Rust with macros)
2. Compile with `hipcc --genco` to `.hsaco` binary
3. Load at runtime with `Module::load()` or `Module::load_data()`
4. Get function with `module.get_function()`
5. Launch with `function.launch()`

**We'll do the same for Candle kernels.**

---

## Week 2: Translation and Build System

### Day 6-7: Translate CUDA to HIP

#### Task 2.1: Create HIP Source Directory

```bash
cd /home/vince/Projects/rbee/deps/candle/candle-kernels

# Create HIP source directory
mkdir -p src/hip

# Create build output directory
mkdir -p hsaco
```

**Checklist:**
- [ ] Created src/hip/
- [ ] Created hsaco/

---

#### Task 2.2: Translate Simple Kernels First

**Priority order (start small):**
1. affine.cu (1.7KB)
2. fill.cu (3.3KB)
3. sort.cu (2.6KB)
4. ternary.cu (2.6KB)

**Translate affine.cu:**

```bash
cd /home/vince/Projects/rbee/deps/candle/candle-kernels

hipify-clang src/affine.cu \
    --cuda-path=/usr/local/cuda \
    -I src \
    -o src/hip/affine.hip

# Review the translation
cat src/hip/affine.hip
```

**Manual review checklist:**
- [ ] `cudaMalloc` → `hipMalloc`
- [ ] `cudaMemcpy` → `hipMemcpy`
- [ ] `__global__` → `__global__` (same)
- [ ] `blockIdx.x` → `hipBlockIdx_x` or stays same
- [ ] `threadIdx.x` → `hipThreadIdx_x` or stays same

**Repeat for fill, sort, ternary:**

```bash
for kernel in fill sort ternary; do
    hipify-clang src/${kernel}.cu \
        --cuda-path=/usr/local/cuda \
        -I src \
        -o src/hip/${kernel}.hip
done
```

**Checklist:**
- [ ] affine.hip created
- [ ] fill.hip created
- [ ] sort.hip created
- [ ] ternary.hip created
- [ ] Manual review done

---

### Day 8: Compile to HSACO

#### Task 2.3: Create Compilation Script

**File:** `candle-kernels/compile_kernels.sh`

```bash
#!/bin/bash
# Created by: TEAM-488 (Phase 2)
# Compile HIP kernels to HSACO binaries

set -e

HIP_DIR="src/hip"
HSACO_DIR="hsaco"

echo "=== Compiling HIP Kernels to HSACO ==="

# Create output directory
mkdir -p "$HSACO_DIR"

# Compile each kernel
for hip_file in "$HIP_DIR"/*.hip; do
    if [ -f "$hip_file" ]; then
        name=$(basename "$hip_file" .hip)
        output="$HSACO_DIR/${name}.hsaco"
        
        echo "Compiling: $name"
        hipcc --genco "$hip_file" -o "$output"
        
        if [ $? -eq 0 ]; then
            echo "✅ $name.hsaco ($(stat -f%z "$output" 2>/dev/null || stat -c%s "$output") bytes)"
        else
            echo "❌ Failed to compile $name"
            exit 1
        fi
    fi
done

echo ""
echo "=== Compilation Complete ==="
ls -lh "$HSACO_DIR"
```

```bash
chmod +x compile_kernels.sh
./compile_kernels.sh
```

**Expected output:**
```
=== Compiling HIP Kernels to HSACO ===
Compiling: affine
✅ affine.hsaco (XXXX bytes)
Compiling: fill
✅ fill.hsaco (XXXX bytes)
...
```

**Checklist:**
- [ ] Script created
- [ ] All kernels compile
- [ ] .hsaco files in hsaco/

---

### Day 9-10: Build System Integration

#### Task 2.4: Embed HSACO in Rust

**File:** `candle-kernels/build.rs`

```rust
// candle-kernels/build.rs
// Created by: TEAM-488 (Phase 2)
// Build system for ROCm kernels

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    #[cfg(feature = "rocm")]
    {
        compile_hip_kernels();
        embed_hsaco_binaries();
    }
}

#[cfg(feature = "rocm")]
fn compile_hip_kernels() {
    println!("cargo:rerun-if-changed=src/hip/");
    
    let hip_dir = PathBuf::from("src/hip");
    let hsaco_dir = PathBuf::from("hsaco");
    
    // Create output directory
    fs::create_dir_all(&hsaco_dir).expect("Failed to create hsaco directory");
    
    // Compile each HIP kernel
    for entry in fs::read_dir(&hip_dir).expect("Failed to read hip directory") {
        let entry = entry.expect("Failed to read entry");
        let path = entry.path();
        
        if path.extension().and_then(|s| s.to_str()) == Some("hip") {
            let name = path.file_stem().unwrap().to_str().unwrap();
            let output = hsaco_dir.join(format!("{}.hsaco", name));
            
            println!("cargo:warning=Compiling HIP kernel: {}", name);
            
            let status = Command::new("hipcc")
                .args(&["--genco", path.to_str().unwrap(), "-o", output.to_str().unwrap()])
                .status()
                .expect("Failed to run hipcc");
            
            if !status.success() {
                panic!("Failed to compile HIP kernel: {}", name);
            }
        }
    }
}

#[cfg(feature = "rocm")]
fn embed_hsaco_binaries() {
    println!("cargo:rerun-if-changed=hsaco/");
    
    let hsaco_dir = PathBuf::from("hsaco");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    let mut embed_code = String::from("// Auto-generated kernel binaries\n");
    embed_code.push_str("// Created by: TEAM-488 (Phase 2)\n\n");
    
    for entry in fs::read_dir(&hsaco_dir).expect("Failed to read hsaco directory") {
        let entry = entry.expect("Failed to read entry");
        let path = entry.path();
        
        if path.extension().and_then(|s| s.to_str()) == Some("hsaco") {
            let name = path.file_stem().unwrap().to_str().unwrap();
            let const_name = name.to_uppercase().replace("-", "_");
            
            // Get absolute path for include_bytes!
            let abs_path = fs::canonicalize(&path).expect("Failed to get absolute path");
            
            embed_code.push_str(&format!(
                "pub const {}_HSACO: &[u8] = include_bytes!(\"{}\");\n",
                const_name,
                abs_path.display()
            ));
        }
    }
    
    fs::write(out_dir.join("kernels.rs"), embed_code)
        .expect("Failed to write kernels.rs");
    
    println!("cargo:warning=Embedded {} kernel binaries", 
        fs::read_dir(&hsaco_dir).unwrap().count());
}
```

**Checklist:**
- [ ] build.rs created
- [ ] Compiles kernels at build time
- [ ] Embeds binaries in Rust

---

#### Task 2.5: Create Kernel Module

**File:** `candle-kernels/src/rocm.rs`

```rust
// candle-kernels/src/rocm.rs
// Created by: TEAM-488 (Phase 2)
// ROCm kernel loading and management

use rocm_rs::hip::{Module, Function};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// Include embedded kernel binaries
include!(concat!(env!("OUT_DIR"), "/kernels.rs"));

/// Kernel cache for efficient loading
pub struct KernelCache {
    modules: Arc<Mutex<HashMap<String, Module>>>,
}

impl KernelCache {
    pub fn new() -> Self {
        Self {
            modules: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Load a kernel by name
    pub fn load(&self, name: &str) -> Result<Function, rocm_rs::hip::Error> {
        let mut modules = self.modules.lock().unwrap();
        
        // Check if already loaded
        if !modules.contains_key(name) {
            // Get binary data
            let binary = match name {
                "affine" => AFFINE_HSACO,
                "fill" => FILL_HSACO,
                "sort" => SORT_HSACO,
                "ternary" => TERNARY_HSACO,
                // Add more as we translate them
                _ => return Err(rocm_rs::hip::Error::new(
                    rocm_rs::hip::ffi::hipError_t_hipErrorInvalidValue
                )),
            };
            
            // Load module from binary
            let module = Module::load_data(binary)?;
            modules.insert(name.to_string(), module);
        }
        
        // Get function from module
        let module = modules.get(name).unwrap();
        module.get_function(&format!("{}_kernel", name))
    }
}

impl Default for KernelCache {
    fn default() -> Self {
        Self::new()
    }
}

// Thread-local kernel cache
thread_local! {
    static KERNEL_CACHE: KernelCache = KernelCache::new();
}

/// Get a kernel function by name
pub fn get_kernel(name: &str) -> Result<Function, rocm_rs::hip::Error> {
    KERNEL_CACHE.with(|cache| cache.load(name))
}
```

**Checklist:**
- [ ] Created rocm.rs
- [ ] Includes embedded binaries
- [ ] KernelCache implemented
- [ ] get_kernel() works

---

### Day 11-12: Test Kernel Loading

#### Task 2.6: Write Kernel Tests

**File:** `candle-kernels/tests/rocm_kernels.rs`

```rust
// tests/rocm_kernels.rs
// Created by: TEAM-488 (Phase 2)

#[cfg(feature = "rocm")]
mod rocm_kernel_tests {
    use candle_kernels::rocm::get_kernel;
    use rocm_rs::hip::{Device, DeviceMemory, Dim3};

    #[test]
    fn test_load_affine_kernel() {
        if !rocm_rs::hip::is_hip_available() {
            return;
        }

        let device = Device::new(0).unwrap();
        device.set_current().unwrap();

        // Load kernel
        let function = get_kernel("affine").expect("Failed to load affine kernel");
        
        println!("✅ Affine kernel loaded successfully");
    }

    #[test]
    fn test_affine_kernel_execution() {
        if !rocm_rs::hip::is_hip_available() {
            return;
        }

        let device = Device::new(0).unwrap();
        device.set_current().unwrap();

        // Allocate memory
        let size = 1024;
        let mut input = DeviceMemory::<f32>::new(size).unwrap();
        let output = DeviceMemory::<f32>::new(size).unwrap();

        // Prepare input data
        let host_input: Vec<f32> = (0..size).map(|i| i as f32).collect();
        input.copy_from_host(&host_input).unwrap();

        // Load and launch kernel
        let function = get_kernel("affine").unwrap();
        
        let grid = Dim3 { x: (size / 256) as u32, y: 1, z: 1 };
        let block = Dim3 { x: 256, y: 1, z: 1 };
        
        let scale = 2.0f32;
        let bias = 1.0f32;
        let size_u32 = size as u32;
        
        let mut args = [
            input.as_kernel_arg(),
            output.as_kernel_arg(),
            &scale as *const _ as *mut std::ffi::c_void,
            &bias as *const _ as *mut std::ffi::c_void,
            &size_u32 as *const _ as *mut std::ffi::c_void,
        ];

        function.launch(grid, block, 0, None, &mut args).unwrap();

        // Copy result back
        let mut host_output = vec![0.0f32; size];
        output.copy_to_host(&mut host_output).unwrap();

        // Verify (y = 2*x + 1)
        for i in 0..size {
            let expected = 2.0 * (i as f32) + 1.0;
            let actual = host_output[i];
            assert!((expected - actual).abs() < 1e-5, 
                "Mismatch at {}: expected {}, got {}", i, expected, actual);
        }

        println!("✅ Affine kernel execution test passed");
    }
}
```

**Run tests:**
```bash
cd /home/vince/Projects/rbee/deps/candle/candle-kernels
cargo test --features rocm rocm_kernel_tests
```

**Checklist:**
- [ ] Kernel loading test passes
- [ ] Kernel execution test passes
- [ ] Results correct

---

## Week 3: Translate Remaining Kernels

### Day 13-14: Medium Complexity Kernels

Translate and test:
- binary.cu (4.9KB)
- cast.cu (7.9KB)
- unary.cu (8.7KB)

**Process for each:**
1. Translate with hipify-clang
2. Manual review
3. Compile to .hsaco
4. Add to kernel cache
5. Write test
6. Verify correctness

**Checklist:**
- [ ] binary.hip translated
- [ ] cast.hip translated
- [ ] unary.hip translated
- [ ] All compile to .hsaco
- [ ] Tests pass

---

### Day 15-16: Complex Kernels

Translate and test:
- indexing.cu (15KB)
- conv.cu (23KB)
- reduce.cu (25KB)

**Special attention:**
- **indexing.cu** - Complex index calculations
- **conv.cu** - Shared memory usage
- **reduce.cu** - Warp-level primitives

**Checklist:**
- [ ] indexing.hip translated
- [ ] conv.hip translated
- [ ] reduce.hip translated
- [ ] All compile
- [ ] Tests pass

---

### Day 17: The Big One - quantized.cu

**File:** quantized.cu (158KB!)

**Strategy:**
1. Translate in sections
2. Review carefully
3. Test each quantization type separately

```bash
hipify-clang src/quantized.cu \
    --cuda-path=/usr/local/cuda \
    -I src \
    -o src/hip/quantized.hip \
    2>&1 | tee quantized_translation.log

# Review log for issues
cat quantized_translation.log
```

**Checklist:**
- [ ] quantized.hip translated
- [ ] Compiles to .hsaco
- [ ] INT8 quantization works
- [ ] INT4 quantization works
- [ ] Tests pass

---

## Final Integration

### Task 2.7: Update Kernel Cache

**File:** `candle-kernels/src/rocm.rs` (update)

```rust
/// Load a kernel by name
pub fn load(&self, name: &str) -> Result<Function, rocm_rs::hip::Error> {
    let mut modules = self.modules.lock().unwrap();
    
    if !modules.contains_key(name) {
        let binary = match name {
            "affine" => AFFINE_HSACO,
            "fill" => FILL_HSACO,
            "sort" => SORT_HSACO,
            "ternary" => TERNARY_HSACO,
            "binary" => BINARY_HSACO,
            "cast" => CAST_HSACO,
            "unary" => UNARY_HSACO,
            "indexing" => INDEXING_HSACO,
            "conv" => CONV_HSACO,
            "reduce" => REDUCE_HSACO,
            "quantized" => QUANTIZED_HSACO,
            _ => return Err(rocm_rs::hip::Error::new(
                rocm_rs::hip::ffi::hipError_t_hipErrorInvalidValue
            )),
        };
        
        let module = Module::load_data(binary)?;
        modules.insert(name.to_string(), module);
    }
    
    let module = modules.get(name).unwrap();
    module.get_function(&format!("{}_kernel", name))
}
```

**Checklist:**
- [ ] All 11 kernels in cache
- [ ] All load correctly

---

## Commit and Push

```bash
cd /home/vince/Projects/rbee/deps/candle

git add candle-kernels/src/hip/
git add candle-kernels/hsaco/
git add candle-kernels/build.rs
git add candle-kernels/src/rocm.rs
git add candle-kernels/tests/rocm_kernels.rs
git add candle-kernels/compile_kernels.sh

git commit -m "TEAM-488: Phase 2 - ROCm kernel compilation system complete

Translated and compiled all 11 CUDA kernels to HIP:

Translation:
- affine, fill, sort, ternary (simple)
- binary, cast, unary (medium)
- indexing, conv, reduce (complex)
- quantized (very complex, 158KB)

Build System:
- build.rs compiles kernels at build time
- Embeds .hsaco binaries in Rust
- KernelCache for efficient loading
- Thread-local caching

All kernels:
- Compile to .hsaco successfully
- Load at runtime
- Execute correctly
- Tests passing

Ready for Phase 3 (backend operations)."

git push origin rocm-support
```

---

## Success Criteria Review

At the end of Phase 2, you should have:

- ✅ All 11 CUDA kernels translated to HIP
- ✅ All kernels compile to `.hsaco` binaries
- ✅ Build system embeds binaries
- ✅ Runtime loading works
- ✅ Basic tests pass

---

## Next Phase

**Phase 3: Backend Operations**

Document: `ROCM_PHASE3_BACKEND_OPERATIONS.md`

---

**Created by:** TEAM-488  
**Date:** 2025-11-13  
**Status:** 📋 PHASE 2 GUIDE
