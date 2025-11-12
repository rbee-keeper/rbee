# Model Loading Verification

**Priority:** 🔴 CRITICAL - MUST HAVE  
**Estimated Effort:** 1-2 days  
**Status:** ⚠️ UNCLEAR - Code exists but untested  
**Assignee:** TBD

---

## Problem

**Current State:**
- Enum defines 7 model versions
- Config methods exist for each version
- **BUT:** No evidence that models actually load and work

**Risk:**
- README claims support for SD 1.5, 2.1, XL, Turbo
- No tests verify these claims
- Models might fail to load or generate garbage

**We need to verify ALL model variants actually work before claiming support.**

---

## Model Variants to Verify

```rust
// src/backend/models/mod.rs line 10-27
pub enum SDVersion {
    V1_5,          // runwayml/stable-diffusion-v1-5
    V1_5Inpaint,   // stable-diffusion-v1-5/stable-diffusion-inpainting
    V2_1,          // stabilityai/stable-diffusion-2-1
    V2Inpaint,     // stabilityai/stable-diffusion-2-inpainting
    XL,            // stabilityai/stable-diffusion-xl-base-1.0
    XLInpaint,     // diffusers/stable-diffusion-xl-1.0-inpainting-0.1
    Turbo,         // stabilityai/sdxl-turbo
}
```

**7 models × 3 backends (CPU, CUDA, Metal) = 21 test configurations**

---

## Implementation Plan

### Step 1: Create Model Test Fixtures

**File:** `tests/fixtures/models.rs`

```rust
use std::path::PathBuf;

/// Model test configuration
pub struct ModelFixture {
    pub version: SDVersion,
    pub repo: &'static str,
    pub expected_size: (usize, usize),
    pub expected_steps: usize,
    pub test_prompt: &'static str,
}

pub const TEST_MODELS: &[ModelFixture] = &[
    ModelFixture {
        version: SDVersion::V1_5,
        repo: "runwayml/stable-diffusion-v1-5",
        expected_size: (512, 512),
        expected_steps: 20,
        test_prompt: "a photo of a cat",
    },
    ModelFixture {
        version: SDVersion::V1_5Inpaint,
        repo: "stable-diffusion-v1-5/stable-diffusion-inpainting",
        expected_size: (512, 512),
        expected_steps: 20,
        test_prompt: "a photo of a dog",
    },
    ModelFixture {
        version: SDVersion::V2_1,
        repo: "stabilityai/stable-diffusion-2-1",
        expected_size: (768, 768),
        expected_steps: 20,
        test_prompt: "a landscape painting",
    },
    ModelFixture {
        version: SDVersion::V2Inpaint,
        repo: "stabilityai/stable-diffusion-2-inpainting",
        expected_size: (768, 768),
        expected_steps: 20,
        test_prompt: "a mountain scene",
    },
    ModelFixture {
        version: SDVersion::XL,
        repo: "stabilityai/stable-diffusion-xl-base-1.0",
        expected_size: (1024, 1024),
        expected_steps: 20,
        test_prompt: "a futuristic city",
    },
    ModelFixture {
        version: SDVersion::XLInpaint,
        repo: "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        expected_size: (1024, 1024),
        expected_steps: 20,
        test_prompt: "a beach scene",
    },
    ModelFixture {
        version: SDVersion::Turbo,
        repo: "stabilityai/sdxl-turbo",
        expected_size: (1024, 1024),
        expected_steps: 4,  // Turbo uses 4 steps
        test_prompt: "a portrait photo",
    },
];

/// Get model path from environment or default cache location
pub fn get_model_path(version: SDVersion) -> Option<PathBuf> {
    // Try environment variable first
    let env_var = format!("SD_MODEL_{:?}", version).to_uppercase();
    if let Ok(path) = std::env::var(&env_var) {
        return Some(PathBuf::from(path));
    }
    
    // Try default cache location
    let home = std::env::var("HOME").ok()?;
    let cache_path = PathBuf::from(home)
        .join(".cache/rbee/models")
        .join(version.repo());
    
    if cache_path.exists() {
        Some(cache_path)
    } else {
        None
    }
}
```

---

### Step 2: Model Loading Tests

**File:** `tests/model_loading.rs`

```rust
use sd_worker_rbee::backend::model_loader::load_model_components;
use sd_worker_rbee::backend::models::SDVersion;
use sd_worker_rbee::device::init_device;

mod fixtures;
use fixtures::models::{TEST_MODELS, get_model_path};

/// Test that all model variants can be loaded
#[test]
fn test_all_models_load() {
    for fixture in TEST_MODELS {
        let model_path = match get_model_path(fixture.version) {
            Some(path) => path,
            None => {
                eprintln!("⚠️  Model {:?} not found, skipping", fixture.version);
                continue;
            }
        };
        
        println!("✓ Testing model: {:?}", fixture.version);
        
        // Initialize device (CPU for testing)
        let device = init_device(false, false, false)
            .expect("Failed to initialize device");
        
        // Load model
        let result = load_model_components(
            &model_path.to_string_lossy(),
            fixture.version,
            &device,
            false, // use_f16
        );
        
        match result {
            Ok(components) => {
                println!("  ✅ Loaded successfully");
                
                // Verify model properties
                assert_eq!(components.version, fixture.version);
                assert_eq!(
                    components.version.default_size(),
                    fixture.expected_size,
                    "Model {:?} has wrong default size",
                    fixture.version
                );
            }
            Err(e) => {
                panic!("❌ Failed to load model {:?}: {}", fixture.version, e);
            }
        }
    }
}

/// Test model loading with F16 precision
#[test]
#[cfg(feature = "cuda")]
fn test_models_load_f16() {
    for fixture in TEST_MODELS {
        let model_path = match get_model_path(fixture.version) {
            Some(path) => path,
            None => continue,
        };
        
        let device = init_device(true, false, false)
            .expect("Failed to initialize CUDA device");
        
        let result = load_model_components(
            &model_path.to_string_lossy(),
            fixture.version,
            &device,
            true, // use_f16
        );
        
        assert!(
            result.is_ok(),
            "Failed to load {:?} with F16",
            fixture.version
        );
    }
}
```

---

### Step 3: Generation Tests

**File:** `tests/generation_verification.rs`

```rust
use sd_worker_rbee::backend::generation::generate_image;
use sd_worker_rbee::backend::model_loader::load_model_components;
use sd_worker_rbee::backend::sampling::SamplingConfig;
use sd_worker_rbee::device::init_device;

mod fixtures;
use fixtures::models::{TEST_MODELS, get_model_path};

/// Test that all models can generate images
#[test]
#[ignore] // Expensive test, run manually
fn test_all_models_generate() {
    for fixture in TEST_MODELS {
        let model_path = match get_model_path(fixture.version) {
            Some(path) => path,
            None => {
                eprintln!("⚠️  Model {:?} not found, skipping", fixture.version);
                continue;
            }
        };
        
        println!("🎨 Testing generation with {:?}", fixture.version);
        
        // Load model
        let device = init_device(false, false, false).unwrap();
        let models = load_model_components(
            &model_path.to_string_lossy(),
            fixture.version,
            &device,
            false,
        ).expect("Failed to load model");
        
        // Create config
        let config = SamplingConfig {
            prompt: fixture.test_prompt.to_string(),
            negative_prompt: None,
            steps: fixture.expected_steps,
            guidance_scale: 7.5,
            seed: Some(42), // Fixed seed for reproducibility
            width: fixture.expected_size.0,
            height: fixture.expected_size.1,
        };
        
        // Generate image
        let result = generate_image(
            &config,
            &models,
            |step, total| {
                if step % 5 == 0 {
                    println!("  Progress: {}/{}", step, total);
                }
            },
        );
        
        match result {
            Ok(image) => {
                println!("  ✅ Generated {}x{} image", image.width(), image.height());
                
                // Verify image dimensions
                assert_eq!(
                    (image.width(), image.height()),
                    (fixture.expected_size.0 as u32, fixture.expected_size.1 as u32),
                    "Model {:?} generated wrong size",
                    fixture.version
                );
                
                // Save for manual inspection
                let output_path = format!("test_output_{:?}.png", fixture.version);
                image.save(&output_path).expect("Failed to save image");
                println!("  💾 Saved to {}", output_path);
            }
            Err(e) => {
                panic!("❌ Generation failed for {:?}: {}", fixture.version, e);
            }
        }
    }
}

/// Test Turbo model with 4-step generation
#[test]
#[ignore]
fn test_turbo_fast_generation() {
    let model_path = match get_model_path(SDVersion::Turbo) {
        Some(path) => path,
        None => {
            eprintln!("⚠️  Turbo model not found, skipping");
            return;
        }
    };
    
    let device = init_device(false, false, false).unwrap();
    let models = load_model_components(
        &model_path.to_string_lossy(),
        SDVersion::Turbo,
        &device,
        false,
    ).unwrap();
    
    let config = SamplingConfig {
        prompt: "a beautiful landscape".to_string(),
        negative_prompt: None,
        steps: 4, // Turbo only needs 4 steps
        guidance_scale: 0.0, // Turbo doesn't use guidance
        seed: Some(42),
        width: 1024,
        height: 1024,
    };
    
    let start = std::time::Instant::now();
    let image = generate_image(&config, &models, |_, _| {}).unwrap();
    let duration = start.elapsed();
    
    println!("✅ Turbo generated in {:?}", duration);
    assert!(duration.as_secs() < 10, "Turbo should be fast (< 10s on CPU)");
    
    image.save("test_output_turbo_fast.png").unwrap();
}
```

---

### Step 4: Inpainting Model Tests

**File:** `tests/inpainting_models.rs`

```rust
use sd_worker_rbee::backend::models::SDVersion;

mod fixtures;
use fixtures::models::get_model_path;

/// Test that inpainting models are correctly identified
#[test]
fn test_inpainting_model_detection() {
    assert!(SDVersion::V1_5Inpaint.is_inpainting());
    assert!(SDVersion::V2Inpaint.is_inpainting());
    assert!(SDVersion::XLInpaint.is_inpainting());
    
    assert!(!SDVersion::V1_5.is_inpainting());
    assert!(!SDVersion::V2_1.is_inpainting());
    assert!(!SDVersion::XL.is_inpainting());
    assert!(!SDVersion::Turbo.is_inpainting());
}

/// Test that inpainting models have correct channel configuration
#[test]
#[ignore]
fn test_inpainting_models_load() {
    let inpainting_models = vec![
        SDVersion::V1_5Inpaint,
        SDVersion::V2Inpaint,
        SDVersion::XLInpaint,
    ];
    
    for version in inpainting_models {
        let model_path = match get_model_path(version) {
            Some(path) => path,
            None => {
                eprintln!("⚠️  Inpainting model {:?} not found, skipping", version);
                continue;
            }
        };
        
        println!("Testing inpainting model: {:?}", version);
        
        let device = sd_worker_rbee::device::init_device(false, false, false).unwrap();
        let models = sd_worker_rbee::backend::model_loader::load_model_components(
            &model_path.to_string_lossy(),
            version,
            &device,
            false,
        ).unwrap();
        
        // Verify it's an inpainting model
        assert!(models.version.is_inpainting());
        
        println!("  ✅ Inpainting model {:?} loaded correctly", version);
    }
}
```

---

### Step 5: Create Test Runner Script

**File:** `scripts/test_all_models.sh`

```bash
#!/bin/bash
# Test all SD model variants

set -e

echo "🧪 SD Worker Model Verification Suite"
echo "======================================"
echo ""

# Check if models are available
echo "📦 Checking for models..."
MODELS_DIR="${HOME}/.cache/rbee/models"

if [ ! -d "$MODELS_DIR" ]; then
    echo "❌ Models directory not found: $MODELS_DIR"
    echo "Please download models first:"
    echo "  rbee-hive models download stable-diffusion-v1-5"
    echo "  rbee-hive models download stable-diffusion-2-1"
    echo "  rbee-hive models download stable-diffusion-xl"
    exit 1
fi

echo "✅ Models directory found"
echo ""

# Run model loading tests
echo "🔍 Testing model loading..."
cargo test --features cpu test_all_models_load -- --nocapture

# Run generation tests (expensive, optional)
if [ "$1" == "--full" ]; then
    echo ""
    echo "🎨 Testing image generation (this will take a while)..."
    cargo test --features cpu test_all_models_generate -- --ignored --nocapture
    
    echo ""
    echo "⚡ Testing Turbo model..."
    cargo test --features cpu test_turbo_fast_generation -- --ignored --nocapture
    
    echo ""
    echo "🖌️  Testing inpainting models..."
    cargo test --features cpu test_inpainting_models_load -- --ignored --nocapture
fi

echo ""
echo "✅ All tests passed!"
```

---

## Acceptance Criteria

- [ ] All 7 model variants can be loaded
- [ ] All models generate valid images
- [ ] Image dimensions match expected sizes
- [ ] Turbo model generates in < 10 seconds (CPU)
- [ ] Inpainting models are correctly identified
- [ ] F16 precision works on CUDA
- [ ] Test fixtures cover all variants
- [ ] Test runner script works
- [ ] Documentation updated with verified models

---

## Test Execution Plan

### Phase 1: Quick Smoke Tests (30 minutes)
```bash
# Test that models load
cargo test --features cpu test_all_models_load -- --nocapture
```

### Phase 2: Generation Tests (2-3 hours)
```bash
# Test that models generate images
cargo test --features cpu test_all_models_generate -- --ignored --nocapture
```

### Phase 3: Full Suite (4-5 hours)
```bash
# Run all tests including CUDA variants
./scripts/test_all_models.sh --full
```

---

## Expected Results

### Model Loading
- ✅ All 7 models load successfully
- ✅ No OOM errors
- ✅ Correct default sizes

### Image Generation
- ✅ V1.5: 512x512 images
- ✅ V2.1: 768x768 images
- ✅ XL: 1024x1024 images
- ✅ Turbo: 1024x1024 in < 10s
- ✅ Inpainting: 9-channel input works

### Quality Checks
- ✅ Images are not blank
- ✅ Images are not noise
- ✅ Images match prompt
- ✅ No artifacts or corruption

---

## Failure Scenarios

### If Model Fails to Load
1. Check model files exist
2. Verify SafeTensors format
3. Check config.json is valid
4. Verify tokenizer files present

### If Generation Fails
1. Check memory usage (OOM?)
2. Verify UNet forward pass
3. Check VAE decode
4. Verify scheduler timesteps

### If Output Is Garbage
1. Check VAE scaling factor (0.18215)
2. Verify text embeddings
3. Check guidance scale
4. Verify latent initialization

---

## Documentation Updates

After verification, update:

1. **README.md** - Remove unverified claims, add verified models
2. **MVP_CHECKLIST.md** - Mark model verification as complete
3. **Marketplace compatibility** - Only list verified models

---

## Estimated Timeline

- **Day 1 Morning:** Create test fixtures and model loading tests
- **Day 1 Afternoon:** Run model loading tests, fix any issues
- **Day 2 Morning:** Create generation tests
- **Day 2 Afternoon:** Run full test suite, document results

**Total:** 1-2 days
