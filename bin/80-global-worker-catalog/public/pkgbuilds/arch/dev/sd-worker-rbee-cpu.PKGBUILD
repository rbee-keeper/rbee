# PKGBUILD for sd-worker-rbee (CPU variant)
# Platform: Arch Linux
# Build: Development (build from source)
# Maintainer: rbee Core Team

pkgname=sd-worker-rbee-cpu
pkgver=0.1.0
pkgrel=1
pkgdesc="Stable Diffusion worker for rbee system (CPU-only, development)"
arch=('x86_64' 'aarch64')
url="https://github.com/rbee-keeper/rbee"
license=('GPL-3.0-or-later')
depends=('gcc')
makedepends=('rust' 'cargo' 'git')
source=("git+https://github.com/rbee-keeper/rbee.git#branch=development")
# Release: For production, use https://github.com/rbee-keeper/rbee/releases/download/v${pkgver}/...
sha256sums=('SKIP')

build() {
    cd "$srcdir/rbee/bin/31_sd_worker_rbee"
    cargo build --release --no-default-features --features cpu
}

package() {
    cd "$srcdir/rbee"
    install -Dm755 "target/release/sd-worker-rbee" \
        "$pkgdir/usr/local/bin/$pkgname"
}

check() {
    cd "$srcdir/rbee/bin/31_sd_worker_rbee"
    cargo test --release --no-default-features --features cpu || true
}
