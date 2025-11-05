#!/usr/bin/env bash
# TEAM-450: Simple build script for new machines
# Just runs the root build commands - Turborepo and Cargo handle the rest!

set -e

echo "🐝 Building rbee monorepo..."
echo ""

# ============================================================================
# DETECT OS
# ============================================================================
detect_os() {
  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if [ -f /etc/arch-release ]; then
      echo "arch"
    elif [ -f /etc/debian_version ]; then
      echo "ubuntu"
    else
      echo "linux"
    fi
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "macos"
  else
    echo "unknown"
  fi
}

OS=$(detect_os)

# ============================================================================
# PREFLIGHT CHECKS - FAIL FAST
# ============================================================================
echo "→ Running preflight checks..."
echo ""

FAILED=0
MISSING_DEPS=()

# Check Node.js
echo "[1/5] Checking Node.js..."
if ! command -v node &> /dev/null; then
  echo "  ✗ node is not installed"
  FAILED=1
  MISSING_DEPS+=("node")
else
  NODE_VERSION=$(node --version)
  echo "  ✓ node $NODE_VERSION"
fi

# Check pnpm
echo "[2/5] Checking pnpm..."
if ! command -v pnpm &> /dev/null; then
  echo "  ✗ pnpm is not installed"
  FAILED=1
  MISSING_DEPS+=("pnpm")
else
  PNPM_VERSION=$(pnpm --version)
  echo "  ✓ pnpm $PNPM_VERSION"
fi

# Check Cargo
echo "[3/5] Checking Cargo..."
if ! command -v cargo &> /dev/null; then
  echo "  ✗ cargo is not installed"
  FAILED=1
  MISSING_DEPS+=("cargo")
else
  CARGO_VERSION=$(cargo --version | cut -d' ' -f2)
  echo "  ✓ cargo $CARGO_VERSION"
fi

# Check wasm-pack
echo "[4/5] Checking wasm-pack..."
if ! command -v wasm-pack &> /dev/null; then
  echo "  ✗ wasm-pack is not installed"
  FAILED=1
  MISSING_DEPS+=("wasm-pack")
else
  WASM_PACK_VERSION=$(wasm-pack --version | cut -d' ' -f2)
  echo "  ✓ wasm-pack $WASM_PACK_VERSION"
fi

# Check for required system libraries (pkg-config)
echo "[5/5] Checking system libraries..."
if command -v pkg-config &> /dev/null; then
  # Check glib-2.0
  if ! pkg-config --exists glib-2.0; then
    echo "  ✗ glib-2.0 development library is not installed"
    FAILED=1
    MISSING_DEPS+=("glib")
  else
    GLIB_VERSION=$(pkg-config --modversion glib-2.0)
    echo "  ✓ glib-2.0 $GLIB_VERSION"
  fi
  
  # Check gdk-3.0
  if ! pkg-config --exists gdk-3.0; then
    echo "  ✗ gdk-3.0 development library is not installed"
    FAILED=1
    MISSING_DEPS+=("gdk")
  else
    GDK_VERSION=$(pkg-config --modversion gdk-3.0)
    echo "  ✓ gdk-3.0 $GDK_VERSION"
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
  echo ""
  
  # OS-specific installation instructions
  case "$OS" in
    arch)
      echo "📦 Arch Linux:"
      for dep in "${MISSING_DEPS[@]}"; do
        case "$dep" in
          node) echo "  • Node.js:   sudo pacman -S nodejs npm" ;;
          pnpm) echo "  • pnpm:      sudo npm install -g pnpm" ;;
          cargo) echo "  • Rust:      sudo pacman -S rustup && rustup default stable" ;;
          wasm-pack) echo "  • wasm-pack: cargo install wasm-pack" ;;
          glib) echo "  • glib-2.0:  sudo pacman -S glib2" ;;
          gdk) echo "  • gdk-3.0:   sudo pacman -S gtk3" ;;
        esac
      done
      ;;
    ubuntu)
      echo "📦 Ubuntu/Debian:"
      for dep in "${MISSING_DEPS[@]}"; do
        case "$dep" in
          node) echo "  • Node.js:   curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash - && sudo apt install -y nodejs" ;;
          pnpm) echo "  • pnpm:      sudo npm install -g pnpm" ;;
          cargo) echo "  • Rust:      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh" ;;
          wasm-pack) echo "  • wasm-pack: cargo install wasm-pack" ;;
          glib) echo "  • glib-2.0:  sudo apt install libglib2.0-dev" ;;
          gdk) echo "  • gdk-3.0:   sudo apt install libgtk-3-dev" ;;
        esac
      done
      ;;
    macos)
      echo "📦 macOS:"
      for dep in "${MISSING_DEPS[@]}"; do
        case "$dep" in
          node) echo "  • Node.js:   brew install node" ;;
          pnpm) echo "  • pnpm:      brew install pnpm" ;;
          cargo) echo "  • Rust:      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh" ;;
          wasm-pack) echo "  • wasm-pack: cargo install wasm-pack" ;;
          glib) echo "  • glib-2.0:  brew install glib" ;;
          gdk) echo "  • gdk-3.0:   brew install gtk+3" ;;
        esac
      done
      ;;
    *)
      echo "📦 Generic (visit official sites):"
      for dep in "${MISSING_DEPS[@]}"; do
        case "$dep" in
          node) echo "  • Node.js:   https://nodejs.org/" ;;
          pnpm) echo "  • pnpm:      npm install -g pnpm" ;;
          cargo) echo "  • Rust:      https://rustup.rs/" ;;
          wasm-pack) echo "  • wasm-pack: cargo install wasm-pack" ;;
          glib) echo "  • glib-2.0:  Install glib2 development package for your OS" ;;
          gdk) echo "  • gdk-3.0:   Install GTK3 development package for your OS" ;;
        esac
      done
      ;;
  esac
  
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
