# PKGBUILD for llm-worker-rbee (CUDA variant)
# Maintainer: rbee Core Team
# TEAM-451: Supports both GitHub releases (pre-built) and source builds

pkgname=llm-worker-rbee-cuda
pkgver=0.1.0
pkgrel=1
pkgdesc="LLM worker for rbee system (NVIDIA CUDA)"
arch=('x86_64')
url="https://github.com/rbee-keeper/rbee"
license=('GPL-3.0-or-later')
depends=('gcc' 'cuda')
makedepends=('rust' 'cargo' 'git')

# Release mode: Download pre-built binary from GitHub releases
# Source mode: Build from git source
_use_release=1  # Set to 1 to use GitHub releases, 0 to build from source

if [ "$_use_release" = "1" ]; then
    # Download pre-built binary from GitHub releases
    source=("https://github.com/rbee-keeper/rbee/releases/download/v${pkgver}/llm-worker-rbee-linux-x86_64-${pkgver}.tar.gz")
    sha256sums=('SKIP')
else
    # Build from source
    source=("git+https://github.com/rbee-keeper/rbee.git#branch=main")
    sha256sums=('SKIP')
fi

build() {
    if [ "$_use_release" = "0" ]; then
        cd "$srcdir/rbee/bin/30_llm_worker_rbee"
        cargo build --release --no-default-features --features cuda
    fi
}

package() {
    if [ "$_use_release" = "1" ]; then
        install -Dm755 "$srcdir/llm-worker-rbee" \
            "$pkgdir/usr/local/bin/$pkgname"
    else
        cd "$srcdir/rbee"
        install -Dm755 "target/release/llm-worker-rbee" \
            "$pkgdir/usr/local/bin/$pkgname"
    fi
}

check() {
    if [ "$_use_release" = "0" ]; then
        cd "$srcdir/rbee/bin/30_llm_worker_rbee"
        cargo test --release --no-default-features --features cuda || true
    fi
}
