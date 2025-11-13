# Step 8: Update Examples

**Estimated Time:** 30 minutes  
**Difficulty:** Easy  
**Dependencies:** Steps 1-7

---

## 🎯 OBJECTIVE

Update all example `Cargo.toml` files to use feature flags.

---

## 📝 FILES TO MODIFY

All `Cargo.toml` files in `candle-examples/examples/*/Cargo.toml`

---

## 🔧 PATTERN TO APPLY

### Before (typical example):
```toml
[dependencies]
candle-core = { path = "../../candle-core" }
candle-nn = { path = "../../candle-nn" }
```

### After:
```toml
[dependencies]
candle-core = { path = "../../candle-core", features = ["cpu"] }
candle-nn = { path = "../../candle-nn", features = ["cpu"] }

[features]
default = ["cpu"]
cpu = ["candle-core/cpu", "candle-nn/cpu"]
cuda = ["candle-core/cuda", "candle-nn/cuda"]
metal = ["candle-core/metal", "candle-nn/metal"]
rocm = ["candle-core/rocm", "candle-nn/rocm"]
```

---

## 📋 EXAMPLES TO UPDATE

### CPU-only examples:
- `mnist`
- `simple-training`
- `tensor-basics`

**Pattern:**
```toml
[dependencies]
candle-core = { path = "../../candle-core", default-features = false, features = ["cpu"] }
```

---

### GPU-capable examples:
- `llama`
- `stable-diffusion`
- `whisper`
- `bert`

**Pattern:**
```toml
[dependencies]
candle-core = { path = "../../candle-core", default-features = false }

[features]
default = ["cpu"]
cpu = ["candle-core/cpu"]
cuda = ["candle-core/cuda"]
metal = ["candle-core/metal"]
rocm = ["candle-core/rocm"]
```

**Usage:**
```bash
# Run with CPU
cargo run --example llama --features cpu

# Run with CUDA
cargo run --example llama --features cuda

# Run with Metal
cargo run --example llama --features metal

# Run with ROCm
cargo run --example llama --features rocm
```

---

### Multi-backend examples:
- `device-comparison`
- `benchmark`

**Pattern:**
```toml
[features]
default = ["all-backends"]
all-backends = ["cpu", "cuda", "metal", "rocm"]
cpu = ["candle-core/cpu"]
cuda = ["candle-core/cuda"]
metal = ["candle-core/metal"]
rocm = ["candle-core/rocm"]
```

---

## 🤖 AUTOMATION SCRIPT

```bash
#!/bin/bash
# update_example_features.sh

EXAMPLES_DIR="candle-examples/examples"

for example in "$EXAMPLES_DIR"/*/; do
    CARGO_TOML="$example/Cargo.toml"
    
    if [ -f "$CARGO_TOML" ]; then
        echo "Updating $CARGO_TOML"
        
        # Add features section if not present
        if ! grep -q "\[features\]" "$CARGO_TOML"; then
            cat >> "$CARGO_TOML" << 'EOF'

[features]
default = ["cpu"]
cpu = ["candle-core/cpu"]
cuda = ["candle-core/cuda"]
metal = ["candle-core/metal"]
rocm = ["candle-core/rocm"]
EOF
        fi
    fi
done

echo "Done! Review changes with: git diff $EXAMPLES_DIR"
```

---

## ✅ VERIFICATION

```bash
# Test CPU-only example
cd candle-examples/examples/mnist
cargo run --no-default-features --features cpu

# Test CUDA example
cd candle-examples/examples/llama
cargo run --features cuda

# Test all examples compile
cd candle-examples
cargo check --all-features
```

---

## 📊 EXAMPLES LIST

Update these `Cargo.toml` files:

- [ ] `examples/mnist/Cargo.toml`
- [ ] `examples/llama/Cargo.toml`
- [ ] `examples/stable-diffusion/Cargo.toml`
- [ ] `examples/whisper/Cargo.toml`
- [ ] `examples/bert/Cargo.toml`
- [ ] `examples/yolo/Cargo.toml`
- [ ] `examples/t5/Cargo.toml`
- [ ] `examples/quantized/Cargo.toml`
- [ ] `examples/simple-training/Cargo.toml`
- [ ] `examples/tensor-basics/Cargo.toml`
- [ ] ... (add all examples)

---

## 📊 PROGRESS TRACKING

- [ ] Identify all example directories
- [ ] Update CPU-only examples
- [ ] Update GPU-capable examples
- [ ] Update multi-backend examples
- [ ] Run verification commands
- [ ] Commit changes

---

## 🎯 NEXT STEP

**Proceed to STEP_9_VERIFICATION.md**

---

**TEAM-501 STEP 8**
