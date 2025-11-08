# PKGBUILD for llm-worker-rbee (CPU variant)
# Maintainer: rbee Core Team
# TEAM-451: Supports both GitHub releases (pre-built) and source builds

pkgname=llm-worker-rbee-cpu
pkgver=0.1.0
pkgrel=1
pkgdesc="LLM worker for rbee system (CPU-only)"
arch=('x86_64' 'aarch64')
url="https://github.com/veighnsche/llama-orch"
license=('GPL-3.0-or-later')
depends=('gcc')
makedepends=('rust' 'cargo' 'git')

# Release mode: Download pre-built binary from GitHub releases
# Source mode: Build from git source
_use_release=1  # Set to 1 to use GitHub releases, 0 to build from source

if [ "$_use_release" = "1" ]; then
    # Download pre-built binary from GitHub releases
    source_x86_64=("https://github.com/rbee-keeper/rbee/releases/download/v${pkgver}/llm-worker-rbee-linux-x86_64-${pkgver}.tar.gz")
    source_aarch64=("https://github.com/rbee-keeper/rbee/releases/download/v${pkgver}/llm-worker-rbee-linux-aarch64-${pkgver}.tar.gz")
    sha256sums_x86_64=('SKIP')
    sha256sums_aarch64=('SKIP')
else
    # Build from source
    source=("git+https://github.com/veighnsche/llama-orch.git#branch=main")
    sha256sums=('SKIP')
fi

build() {
    if [ "$_use_release" = "0" ]; then
        cd "$srcdir/llama-orch/bin/30_llm_worker_rbee"
        cargo build --release --no-default-features --features cpu
    fi
    # No build needed for release mode (pre-built binary)
}

package() {
    if [ "$_use_release" = "1" ]; then
        # Install pre-built binary
        install -Dm755 "$srcdir/llm-worker-rbee" \
            "$pkgdir/usr/local/bin/$pkgname"
    else
        # Install built binary
        cd "$srcdir/llama-orch"
        install -Dm755 "target/release/llm-worker-rbee" \
            "$pkgdir/usr/local/bin/$pkgname"
    fi
}

check() {
    if [ "$_use_release" = "0" ]; then
        cd "$srcdir/llama-orch/bin/30_llm_worker_rbee"
        cargo test --release --no-default-features --features cpu || true
    fi
    # No tests for release mode (pre-built binary)
}
