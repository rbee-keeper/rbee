# PKGBUILD for sd-worker-rbee (CUDA variant)
# Maintainer: rbee Core Team
# TEAM-451: Supports both GitHub releases (pre-built) and source builds

pkgname=sd-worker-rbee-cuda
pkgver=0.1.0
pkgrel=1
pkgdesc="Stable Diffusion worker for rbee system (NVIDIA CUDA)"
arch=('x86_64')
url="https://github.com/rbee-keeper/rbee"
license=('GPL-3.0-or-later')
depends=('gcc' 'cuda')
makedepends=('rust' 'cargo' 'git')

_use_release=1

if [ "$_use_release" = "1" ]; then
    source=("https://github.com/rbee-keeper/rbee/releases/download/v${pkgver}/sd-worker-rbee-linux-x86_64-${pkgver}.tar.gz")
    sha256sums=('SKIP')
else
    source=("git+https://github.com/rbee-keeper/rbee.git#branch=main")
    sha256sums=('SKIP')
fi

build() {
    if [ "$_use_release" = "0" ]; then
        cd "$srcdir/rbee/bin/31_sd_worker_rbee"
        cargo build --release --no-default-features --features cuda
    fi
}

package() {
    if [ "$_use_release" = "1" ]; then
        install -Dm755 "$srcdir/sd-worker-rbee" \
            "$pkgdir/usr/local/bin/$pkgname"
    else
        cd "$srcdir/rbee"
        install -Dm755 "target/release/sd-worker-rbee" \
            "$pkgdir/usr/local/bin/$pkgname"
    fi
}

check() {
    if [ "$_use_release" = "0" ]; then
        cd "$srcdir/rbee/bin/31_sd_worker_rbee"
        cargo test --release --no-default-features --features cuda || true
    fi
}
