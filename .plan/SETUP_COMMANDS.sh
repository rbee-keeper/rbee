#!/bin/bash
# Candle Submodule Setup - Automated Script
# Run from: /home/vince/Projects/rbee

set -e  # Exit on error

echo "🚀 Setting up Candle as submodule..."
echo ""

# Phase 1: Verify fork exists
echo "📋 Phase 1: Checking your fork..."
if ! git ls-remote https://github.com/veighnsche/candle.git &>/dev/null; then
    echo "❌ ERROR: Fork not found at https://github.com/veighnsche/candle.git"
    echo "   Please fork https://github.com/huggingface/candle first"
    exit 1
fi
echo "✅ Fork exists"
echo ""

# Phase 2: Backup and remove old reference/
echo "📋 Phase 2: Removing old reference/ directory..."
if [ -d "reference" ]; then
    echo "   Backing up to reference.backup..."
    cp -r reference reference.backup
    echo "   Removing reference/..."
    rm -rf reference/
    echo "✅ Old reference/ removed"
else
    echo "✅ No reference/ directory found (already clean)"
fi
echo ""

# Phase 3: Update .gitignore
echo "📋 Phase 3: Updating .gitignore..."
if grep -q "^/reference/$" .gitignore; then
    echo "   Removing /reference/ from .gitignore..."
    sed -i '/^\/reference\/$/d' .gitignore
    sed -i '/^# Reference directory/d' .gitignore
    echo "✅ .gitignore updated"
else
    echo "✅ .gitignore already clean"
fi
echo ""

# Phase 4: Create deps/ and add submodule
echo "📋 Phase 4: Adding Candle submodule..."
mkdir -p deps

if [ -d "deps/candle" ]; then
    echo "⚠️  deps/candle already exists, skipping..."
else
    echo "   Adding submodule (this may take a minute)..."
    git submodule add -b rocm-support \
        https://github.com/veighnsche/candle.git \
        deps/candle
    
    echo "   Initializing submodule..."
    git submodule update --init --recursive
    
    echo "✅ Submodule added"
fi
echo ""

# Phase 5: Update Cargo.toml
echo "📋 Phase 5: Updating Cargo.toml..."
CARGO_FILE="bin/30_llm_worker_rbee/Cargo.toml"

if grep -q "reference/candle" "$CARGO_FILE"; then
    echo "   Updating paths from reference/ to deps/..."
    sed -i 's|../../reference/candle|../../deps/candle|g' "$CARGO_FILE"
    echo "✅ Cargo.toml updated"
else
    echo "✅ Cargo.toml already uses deps/"
fi
echo ""

# Phase 6: Verify setup
echo "📋 Phase 6: Verifying setup..."

echo "   Checking submodule status..."
git submodule status deps/candle

echo "   Checking Cargo.toml paths..."
if grep -q "deps/candle" "$CARGO_FILE"; then
    echo "   ✅ Cargo.toml paths correct"
else
    echo "   ❌ ERROR: Cargo.toml paths not updated"
    exit 1
fi

echo "   Checking submodule files..."
if [ -f "deps/candle/candle-core/Cargo.toml" ]; then
    echo "   ✅ Submodule files present"
else
    echo "   ❌ ERROR: Submodule files missing"
    exit 1
fi
echo ""

# Phase 7: Test build
echo "📋 Phase 7: Testing build..."
echo "   Running cargo check (this may take a minute)..."
cd bin/30_llm_worker_rbee
if cargo check --bin llm-worker-rbee 2>&1 | tail -5; then
    echo "✅ Build successful!"
else
    echo "⚠️  Build had warnings (check output above)"
fi
cd ../..
echo ""

# Phase 8: Show next steps
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Review changes: git status"
echo "   2. Commit changes:"
echo "      git add .gitignore .gitmodules deps/candle bin/30_llm_worker_rbee/Cargo.toml"
echo "      git commit -m 'Add Candle as submodule for ROCm development'"
echo "      git push"
echo ""
echo "   3. Start ROCm development:"
echo "      cd deps/candle"
echo "      git checkout rocm-support"
echo "      # Make your changes!"
echo ""
echo "📚 See .plan/CANDLE_SUBMODULE_SETUP.md for full documentation"
