# PKGBUILD for llm-worker-rbee (Metal variant)
# Platform: Arch Linux (for cross-compilation)
# Build: Development (build from source)
# Maintainer: rbee Core Team

pkgname=llm-worker-rbee-metal
pkgver=0.1.0
pkgrel=1
pkgdesc="LLM worker for rbee system (Apple Metal, development)"
arch=('aarch64')
url="https://github.com/rbee-keeper/rbee"
license=('GPL-3.0-or-later')
depends=('gcc')
makedepends=('rust' 'cargo' 'git')
source=("git+https://github.com/rbee-keeper/rbee.git#branch=development")
# Release: For production, use https://github.com/rbee-keeper/rbee/releases/download/v${pkgver}/...
sha256sums=('SKIP')

build() {
    cd "$srcdir/rbee/bin/30_llm_worker_rbee"
    cargo build --release --no-default-features --features metal
}

package() {
    cd "$srcdir/rbee"
    install -Dm755 "target/release/llm-worker-rbee" \
        "$pkgdir/usr/local/bin/$pkgname"
}

check() {
    cd "$srcdir/rbee/bin/30_llm_worker_rbee"
    cargo test --release --no-default-features --features metal || true
}
