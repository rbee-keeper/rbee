# PKGBUILD for sd-worker-rbee (Metal variant)
# Maintainer: rbee Core Team
# TEAM-403: Stable Diffusion worker with Apple Metal acceleration

pkgname=sd-worker-rbee-metal
pkgver=0.1.0
pkgrel=1
pkgdesc="Stable Diffusion worker for rbee system (Apple Metal)"
arch=('aarch64')
url="https://github.com/veighnsche/llama-orch"
license=('GPL-3.0-or-later')
depends=('clang')
makedepends=('rust' 'cargo' 'git')
source=("git+https://github.com/veighnsche/llama-orch.git#branch=main")
sha256sums=('SKIP')

build() {
    cd "$srcdir/llama-orch/bin/31_sd_worker_rbee"
    cargo build --release --no-default-features --features metal
}

package() {
    cd "$srcdir/llama-orch"
    # TEAM-403: Use workspace-level target directory (Cargo workspace outputs to root target/)
    install -Dm755 "target/release/sd-worker-rbee" \
        "$pkgdir/usr/local/bin/$pkgname"
}

check() {
    cd "$srcdir/llama-orch/bin/31_sd_worker_rbee"
    cargo test --release --no-default-features --features metal || true
}
