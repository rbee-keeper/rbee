#!/usr/bin/env bash
# TEAM-450: Simple build script for new machines
# Just runs the root build commands - Turborepo and Cargo handle the rest!

set -e

echo "🐝 Building rbee monorepo..."
echo ""

# ============================================================================
# PREFLIGHT CHECKS - FAIL FAST
# ============================================================================
echo "→ Running preflight checks..."
echo ""

FAILED=0

# Check Node.js
echo "[1/5] Checking Node.js..."
if ! command -v node &> /dev/null; then
  echo "  ✗ node is not installed"
  FAILED=1
else
  NODE_VERSION=$(node --version)
  echo "  ✓ node $NODE_VERSION"
fi

# Check pnpm
echo "[2/5] Checking pnpm..."
if ! command -v pnpm &> /dev/null; then
  echo "  ✗ pnpm is not installed"
  FAILED=1
else
  PNPM_VERSION=$(pnpm --version)
  echo "  ✓ pnpm $PNPM_VERSION"
fi

# Check Cargo
echo "[3/5] Checking Cargo..."
if ! command -v cargo &> /dev/null; then
  echo "  ✗ cargo is not installed"
  FAILED=1
else
  CARGO_VERSION=$(cargo --version | cut -d' ' -f2)
  echo "  ✓ cargo $CARGO_VERSION"
fi

# Check wasm-pack
echo "[4/5] Checking wasm-pack..."
if ! command -v wasm-pack &> /dev/null; then
  echo "  ✗ wasm-pack is not installed"
  FAILED=1
else
  WASM_PACK_VERSION=$(wasm-pack --version | cut -d' ' -f2)
  echo "  ✓ wasm-pack $WASM_PACK_VERSION"
fi

# Check for required system libraries (pkg-config)
echo "[5/5] Checking system libraries..."
if command -v pkg-config &> /dev/null; then
  if ! pkg-config --exists glib-2.0; then
    echo "  ✗ glib-2.0 development library is not installed"
    echo "  Install with: sudo apt install libglib2.0-dev"
    FAILED=1
  else
    GLIB_VERSION=$(pkg-config --modversion glib-2.0)
    echo "  ✓ glib-2.0 $GLIB_VERSION"
  fi
else
  echo "  ⚠ pkg-config not found - skipping system library checks"
fi

echo ""

# Exit if any checks failed
if [ $FAILED -eq 1 ]; then
  echo "✗ Preflight checks failed!"
  echo ""
  echo "Install missing dependencies:"
  echo "  • Node.js:   https://nodejs.org/"
  echo "  • pnpm:      npm install -g pnpm"
  echo "  • Rust:      https://rustup.rs/"
  echo "  • wasm-pack: cargo install wasm-pack"
  echo "  • glib-2.0:  sudo apt install libglib2.0-dev"
  echo ""
  exit 1
fi

echo "✓ All preflight checks passed!"
echo ""

# ============================================================================
# BUILD
# ============================================================================

# Install dependencies
echo "→ [BUILD 1/3] Installing dependencies..."
if ! pnpm install; then
  echo "✗ pnpm install failed!"
  exit 1
fi
echo "  ✓ Dependencies installed"
echo ""

# Build frontend (Turborepo handles everything)
echo "→ [BUILD 2/3] Building frontend (Turborepo)..."
if ! pnpm run build; then
  echo "✗ Frontend build failed!"
  exit 1
fi
echo "  ✓ Frontend built"
echo ""

# Build Rust (Cargo workspace handles everything)
echo "→ [BUILD 3/3] Building Rust (Cargo)..."
if ! cargo build --release; then
  echo "✗ Rust build failed!"
  exit 1
fi
echo "  ✓ Rust built"
echo ""

echo "✓ Build complete! 🐝"
