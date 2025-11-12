#!/bin/bash
# TEAM-487: Test all SD model variants
#
# Usage:
#   ./scripts/test_all_models.sh          # Quick smoke tests
#   ./scripts/test_all_models.sh --full   # Full test suite (expensive)

set -e

echo "🧪 SD Worker Model Verification Suite"
echo "======================================"
echo ""

# Check if models are available
echo "📦 Checking for models..."
MODELS_DIR="${HOME}/.cache/rbee/models"

if [ ! -d "$MODELS_DIR" ]; then
    echo "⚠️  Models directory not found: $MODELS_DIR"
    echo ""
    echo "To download models, use one of these methods:"
    echo ""
    echo "  1. Using rbee-hive (recommended):"
    echo "     rbee-hive models download stable-diffusion-v1-5"
    echo "     rbee-hive models download stable-diffusion-2-1"
    echo "     rbee-hive models download stable-diffusion-xl"
    echo ""
    echo "  2. Using environment variables:"
    echo "     export SD_MODEL_V1_5=/path/to/sd-v1-5"
    echo "     export SD_MODEL_V2_1=/path/to/sd-v2-1"
    echo ""
    echo "  3. Manual download to:"
    echo "     ~/.cache/rbee/models/runwayml/stable-diffusion-v1-5/"
    echo "     ~/.cache/rbee/models/stabilityai/stable-diffusion-2-1/"
    echo ""
    echo "Tests will skip models that aren't found."
    echo ""
fi

# Change to project directory
cd "$(dirname "$0")/.."

# Run model config tests (always fast)
echo "🔍 Testing model configurations..."
cargo test --test model_loading test_model_configs -- --nocapture
echo ""

# Run inpainting detection tests (fast)
echo "🖌️  Testing inpainting model detection..."
cargo test --test inpainting_models test_inpainting_model_detection -- --nocapture
cargo test --test inpainting_models test_xl_model_detection -- --nocapture
echo ""

# Run model loading tests (quick smoke test)
echo "📦 Testing model loading..."
cargo test --test model_loading test_all_models_load -- --nocapture
echo ""

# Run full generation tests if requested
if [ "$1" == "--full" ]; then
    echo "🎨 Running full generation tests (this will take a while)..."
    echo ""
    
    echo "📸 Testing image generation..."
    cargo test --test generation_verification test_all_models_generate -- --ignored --nocapture
    echo ""
    
    echo "⚡ Testing Turbo model..."
    cargo test --test generation_verification test_turbo_fast_generation -- --ignored --nocapture
    echo ""
    
    echo "📐 Testing custom sizes..."
    cargo test --test generation_verification test_custom_sizes -- --ignored --nocapture
    echo ""
    
    echo "🖌️  Testing inpainting models..."
    cargo test --test inpainting_models test_inpainting_models_load -- --ignored --nocapture
    echo ""
    
    echo "🚫 Testing non-inpainting model rejection..."
    cargo test --test inpainting_models test_non_inpainting_model_rejects_inpaint -- --nocapture
    echo ""
    
    echo "📊 Test outputs saved to:"
    ls -lh test_output_*.png 2>/dev/null || echo "   (no output images generated)"
    echo ""
fi

echo "✅ All tests passed!"
echo ""

if [ "$1" != "--full" ]; then
    echo "💡 Tip: Run './scripts/test_all_models.sh --full' for complete generation tests"
fi
