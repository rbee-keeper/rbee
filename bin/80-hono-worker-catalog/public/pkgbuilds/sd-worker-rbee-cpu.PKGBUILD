# PKGBUILD for sd-worker-rbee (CPU variant)
# Maintainer: rbee Core Team
# TEAM-451: Supports both GitHub releases (pre-built) and source builds

pkgname=sd-worker-rbee-cpu
pkgver=0.1.0
pkgrel=1
pkgdesc="Stable Diffusion worker for rbee system (CPU-only)"
arch=('x86_64' 'aarch64')
url="https://github.com/veighnsche/llama-orch"
license=('GPL-3.0-or-later')
depends=('gcc')
makedepends=('rust' 'cargo' 'git')

_use_release=1

if [ "$_use_release" = "1" ]; then
    source_x86_64=("https://github.com/rbee-keeper/rbee/releases/download/v${pkgver}/sd-worker-rbee-linux-x86_64-${pkgver}.tar.gz")
    source_aarch64=("https://github.com/rbee-keeper/rbee/releases/download/v${pkgver}/sd-worker-rbee-linux-aarch64-${pkgver}.tar.gz")
    sha256sums_x86_64=('SKIP')
    sha256sums_aarch64=('SKIP')
else
    source=("git+https://github.com/veighnsche/llama-orch.git#branch=main")
    sha256sums=('SKIP')
fi

build() {
    if [ "$_use_release" = "0" ]; then
        cd "$srcdir/llama-orch/bin/31_sd_worker_rbee"
        cargo build --release --no-default-features --features cpu
    fi
}

package() {
    if [ "$_use_release" = "1" ]; then
        install -Dm755 "$srcdir/sd-worker-rbee" \
            "$pkgdir/usr/local/bin/$pkgname"
    else
        cd "$srcdir/llama-orch"
        install -Dm755 "target/release/sd-worker-rbee" \
            "$pkgdir/usr/local/bin/$pkgname"
    fi
}

check() {
    if [ "$_use_release" = "0" ]; then
        cd "$srcdir/llama-orch/bin/31_sd_worker_rbee"
        cargo test --release --no-default-features --features cpu || true
    fi
}
