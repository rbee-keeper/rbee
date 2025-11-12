// TEAM-487: Generation verification tests
//
// Tests that all models can actually generate images (expensive tests)

use sd_worker_rbee::backend::generation::generate_image;
use sd_worker_rbee::backend::model_loader::load_model;
use sd_worker_rbee::backend::models::SDVersion;
use sd_worker_rbee::backend::sampling::SamplingConfig;
use shared_worker_rbee::device::init_cpu_device;

mod fixtures;
use fixtures::models::{get_model_path, TEST_MODELS};

/// Test that all models can generate images
///
/// TEAM-487: Full generation test (expensive, run manually)
/// Run with: cargo test --test generation_verification test_all_models_generate -- --ignored --nocapture
#[test]
#[ignore] // Expensive test, run manually
fn test_all_models_generate() {
    let mut generated_count = 0;
    let mut skipped_count = 0;

    println!("\n🎨 Testing Image Generation");
    println!("============================\n");

    for fixture in TEST_MODELS {
        let model_path = match get_model_path(fixture.version) {
            Some(path) => path,
            None => {
                eprintln!("⚠️  Model {:?} not found, skipping", fixture.version);
                skipped_count += 1;
                continue;
            }
        };

        println!("🖼️  Testing generation with {:?}", fixture.version);
        println!("   Prompt: \"{}\"", fixture.test_prompt);

        // Load model
        let device = init_cpu_device().unwrap();
        let models = load_model(
            fixture.version,
            &device,
            false,
            &[], // TEAM-487: No LoRAs for basic test
            false, // TEAM-483: Not quantized
        )
        .expect("Failed to load model");

        // Create config
        let config = SamplingConfig {
            prompt: fixture.test_prompt.to_string(),
            negative_prompt: None,
            steps: fixture.expected_steps,
            guidance_scale: 7.5,
            seed: Some(42),
            loras: vec![], // Fixed seed for reproducibility
            width: fixture.expected_size.0,
            height: fixture.expected_size.1,
        };

        // Generate image
        let start = std::time::Instant::now();
        let result = generate_image(&config, &models, |step, total, _preview| {
            if step % 5 == 0 || step == total {
                println!("   Progress: {}/{} ({:.1}%)", step, total, (step as f32 / total as f32) * 100.0);
            }
        });
        let duration = start.elapsed();

        match result {
            Ok(image) => {
                println!("   ✅ Generated {}x{} image in {:?}", image.width(), image.height(), duration);

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
                println!("   💾 Saved to {}", output_path);
                println!();

                generated_count += 1;
            }
            Err(e) => {
                panic!("❌ Generation failed for {:?}: {}", fixture.version, e);
            }
        }
    }

    println!("📊 Summary:");
    println!("   Generated: {}", generated_count);
    println!("   Skipped: {}", skipped_count);
    println!("   Total: {}", TEST_MODELS.len());

    if generated_count == 0 {
        panic!("❌ No images were generated!");
    }
}

/// Test Turbo model with 4-step generation
///
/// TEAM-487: Verify Turbo is fast
/// Run with: cargo test --test generation_verification test_turbo_fast_generation -- --ignored --nocapture
#[test]
#[ignore]
fn test_turbo_fast_generation() {
    println!("\n⚡ Testing Turbo Model");
    println!("=====================\n");

    let model_path = match get_model_path(SDVersion::Turbo) {
        Some(path) => path,
        None => {
            eprintln!("⚠️  Turbo model not found, skipping");
            return;
        }
    };

    let device = init_cpu_device().unwrap();
    let models = load_model(SDVersion::Turbo, &device, false, &[], false).unwrap(); // TEAM-487: No LoRAs, TEAM-483: Not quantized

    let config = SamplingConfig {
        prompt: "a beautiful landscape".to_string(),
        negative_prompt: None,
        steps: 4, // Turbo only needs 4 steps
        guidance_scale: 0.0, // Turbo doesn't use guidance
        seed: Some(42),
        width: 1024,
        height: 1024,
            loras: vec![],
    };

    println!("🎨 Generating with Turbo (4 steps, no guidance)...");
    let start = std::time::Instant::now();
    let image = generate_image(&config, &models, |step, total, _preview| {
        println!("   Step {}/{}", step, total);
    })
    .unwrap();
    let duration = start.elapsed();

    println!("✅ Turbo generated 1024x1024 in {:?}", duration);
    println!("   Speed: {:.2} seconds per step", duration.as_secs_f32() / 4.0);

    // Turbo should be reasonably fast even on CPU
    assert!(
        duration.as_secs() < 60,
        "Turbo should generate in < 60s on CPU, took {:?}",
        duration
    );

    image.save("test_output_turbo_fast.png").unwrap();
    println!("💾 Saved to test_output_turbo_fast.png");
}

/// Test generation with different sizes
///
/// TEAM-487: Verify models work with non-default sizes
/// Run with: cargo test --test generation_verification test_custom_sizes -- --ignored --nocapture
#[test]
#[ignore]
fn test_custom_sizes() {
    println!("\n📐 Testing Custom Sizes");
    println!("=======================\n");

    // Test V1.5 with 768x768 (non-default)
    let model_path = match get_model_path(SDVersion::V1_5) {
        Some(path) => path,
        None => {
            eprintln!("⚠️  V1.5 model not found, skipping");
            return;
        }
    };

    let device = init_cpu_device().unwrap();
    let models = load_model(SDVersion::V1_5, &device, false, &[], false).unwrap(); // TEAM-487: No LoRAs, TEAM-483: Not quantized

    let config = SamplingConfig {
        prompt: "a test image".to_string(),
        negative_prompt: None,
        steps: 10, // Fewer steps for speed
        guidance_scale: 7.5,
        seed: Some(42),
        width: 768,  // Non-default size
        height: 768,
            loras: vec![],
    };

    println!("🎨 Generating 768x768 with V1.5 (default is 512x512)...");
    let image = generate_image(&config, &models, |_, _, _| {}).unwrap();

    assert_eq!((image.width(), image.height()), (768, 768));
    println!("✅ Generated custom size successfully");

    image.save("test_output_custom_size.png").unwrap();
    println!("💾 Saved to test_output_custom_size.png");
}
