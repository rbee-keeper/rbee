// TEAM-487: Inpainting model verification tests
//
// Tests that inpainting models are correctly identified and work

use sd_worker_rbee::backend::models::SDVersion;

mod fixtures;
use fixtures::models::get_model_path;

/// Test that inpainting models are correctly identified
///
/// TEAM-487: Verify is_inpainting() method
#[test]
fn test_inpainting_model_detection() {
    println!("\n🖌️  Testing Inpainting Model Detection");
    println!("======================================\n");

    // Inpainting models
    assert!(SDVersion::V1_5Inpaint.is_inpainting());
    println!("✅ V1_5Inpaint correctly identified as inpainting");

    assert!(SDVersion::V2Inpaint.is_inpainting());
    println!("✅ V2Inpaint correctly identified as inpainting");

    assert!(SDVersion::XLInpaint.is_inpainting());
    println!("✅ XLInpaint correctly identified as inpainting");

    // Non-inpainting models
    assert!(!SDVersion::V1_5.is_inpainting());
    println!("✅ V1_5 correctly identified as NOT inpainting");

    assert!(!SDVersion::V2_1.is_inpainting());
    println!("✅ V2_1 correctly identified as NOT inpainting");

    assert!(!SDVersion::XL.is_inpainting());
    println!("✅ XL correctly identified as NOT inpainting");

    assert!(!SDVersion::Turbo.is_inpainting());
    println!("✅ Turbo correctly identified as NOT inpainting");

    println!("\n📊 All 7 models correctly classified");
}

/// Test that inpainting models load correctly
///
/// TEAM-487: Verify inpainting models can be loaded
/// Run with: cargo test --test inpainting_models test_inpainting_models_load -- --ignored --nocapture
#[test]
#[ignore]
fn test_inpainting_models_load() {
    println!("\n🖌️  Testing Inpainting Model Loading");
    println!("====================================\n");

    let inpainting_models = vec![
        SDVersion::V1_5Inpaint,
        SDVersion::V2Inpaint,
        SDVersion::XLInpaint,
    ];

    let mut loaded_count = 0;

    for version in inpainting_models {
        let model_path = match get_model_path(version) {
            Some(path) => path,
            None => {
                eprintln!("⚠️  Inpainting model {:?} not found, skipping", version);
                continue;
            }
        };

        println!("📦 Testing inpainting model: {:?}", version);
        println!("   Path: {}", model_path.display());

        let device = shared_worker_rbee::device::init_cpu_device().unwrap();
        let models = sd_worker_rbee::backend::model_loader::load_model(
            version,
            &device,
            false,
            &[], // TEAM-487: No LoRAs for inpainting test
        )
        .unwrap();

        // Verify it's an inpainting model
        assert!(models.version.is_inpainting());
        println!("   ✅ Loaded and verified as inpainting model");
        println!("   ✓ Default size: {:?}", models.version.default_size());
        println!();

        loaded_count += 1;
    }

    println!("📊 Loaded {} inpainting models", loaded_count);

    if loaded_count == 0 {
        panic!("❌ No inpainting models were loaded!");
    }
}

/// Test that regular models reject inpainting operations
///
/// TEAM-487: Verify error handling for wrong model type
#[test]
fn test_non_inpainting_model_rejects_inpaint() {
    use sd_worker_rbee::backend::generation::inpaint;
    use sd_worker_rbee::backend::model_loader::load_model;
    use sd_worker_rbee::backend::sampling::SamplingConfig;
    use shared_worker_rbee::device::init_cpu_device;
    use image::{DynamicImage, RgbImage};

    println!("\n🚫 Testing Non-Inpainting Model Rejection");
    println!("==========================================\n");

    // Try to load V1.5 (non-inpainting)
    let model_path = match get_model_path(SDVersion::V1_5) {
        Some(path) => path,
        None => {
            eprintln!("⚠️  V1.5 model not found, skipping test");
            return;
        }
    };

    let device = init_cpu_device().unwrap();
    let models = load_model(SDVersion::V1_5, &device, false, &[]).unwrap(); // TEAM-487: No LoRAs

    // Create dummy image and mask
    let input_image = DynamicImage::ImageRgb8(RgbImage::new(512, 512));
    let mask = DynamicImage::ImageRgb8(RgbImage::new(512, 512));

    let config = SamplingConfig {
        prompt: "test".to_string(),
        negative_prompt: None,
        steps: 1,
        guidance_scale: 7.5,
        seed: Some(42),
        width: 512,
        height: 512,
            loras: vec![],
    };

    // Try to inpaint with non-inpainting model
    let result = inpaint(&config, &models, &input_image, &mask, |_, _, _| {});

    // Should fail with error message
    assert!(result.is_err(), "Non-inpainting model should reject inpaint operation");

    let error_msg = result.unwrap_err().to_string();
    assert!(
        error_msg.contains("not an inpainting model"),
        "Error should mention inpainting model requirement, got: {}",
        error_msg
    );

    println!("✅ Non-inpainting model correctly rejected inpaint operation");
    println!("   Error: {}", error_msg);
}

/// Test XL vs non-XL model detection
///
/// TEAM-487: Verify is_xl() method
#[test]
fn test_xl_model_detection() {
    println!("\n🔍 Testing XL Model Detection");
    println!("==============================\n");

    // XL models
    assert!(SDVersion::XL.is_xl());
    println!("✅ XL correctly identified as XL");

    assert!(SDVersion::XLInpaint.is_xl());
    println!("✅ XLInpaint correctly identified as XL");

    assert!(SDVersion::Turbo.is_xl());
    println!("✅ Turbo correctly identified as XL");

    // Non-XL models
    assert!(!SDVersion::V1_5.is_xl());
    println!("✅ V1_5 correctly identified as NOT XL");

    assert!(!SDVersion::V1_5Inpaint.is_xl());
    println!("✅ V1_5Inpaint correctly identified as NOT XL");

    assert!(!SDVersion::V2_1.is_xl());
    println!("✅ V2_1 correctly identified as NOT XL");

    assert!(!SDVersion::V2Inpaint.is_xl());
    println!("✅ V2Inpaint correctly identified as NOT XL");

    println!("\n📊 All XL classifications correct");
}
