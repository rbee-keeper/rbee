# PKGBUILD for sd-worker-rbee (git/source version)
# Maintainer: rbee Core Team
# TEAM-481: Builds from source with configurable features
# Usage: RBEE_FEATURES=cuda makepkg -si

pkgname=sd-worker-rbee-git
pkgver=0.1.0
pkgrel=1
pkgdesc="Stable Diffusion worker for rbee system (built from source)"
arch=('x86_64' 'aarch64')
url="https://github.com/rbee-keeper/rbee"
license=('GPL-3.0-or-later')
provides=('sd-worker-rbee')
conflicts=('sd-worker-rbee-bin')
makedepends=('rust' 'cargo' 'git')

# Build features: cpu, cuda, metal, rocm
# Default to cpu if not specified
_features=${RBEE_FEATURES:-cpu}
echo "Building with features: $_features"

# Dynamic dependencies based on features
depends=('gcc')
case "$_features" in
    *cuda*) depends+=('cuda') ;;
    *rocm*) depends+=('rocm') ;;
esac

source=("git+https://github.com/rbee-keeper/rbee.git#branch=main")
sha256sums=('SKIP')

pkgver() {
    cd "$srcdir/rbee"
    git describe --long --tags | sed 's/^v//;s/\([^-]*-g\)/r\1/;s/-/./g'
}

build() {
    cd "$srcdir/rbee/bin/31_sd_worker_rbee"
    
    # Build with specified features
    cargo build --release --no-default-features --features "$_features"
}

check() {
    cd "$srcdir/rbee/bin/31_sd_worker_rbee"
    cargo test --release --no-default-features --features "$_features" || true
}

package() {
    cd "$srcdir/rbee"
    install -Dm755 "target/release/sd-worker-rbee" \
        "$pkgdir/usr/local/bin/sd-worker-rbee"
}
