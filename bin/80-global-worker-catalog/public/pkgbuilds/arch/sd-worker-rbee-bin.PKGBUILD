# PKGBUILD for sd-worker-rbee (binary/release version)
# Maintainer: rbee Core Team
# TEAM-481: Downloads pre-built binary from GitHub releases
# Auto-detects platform: linux (CUDA/ROCm/CPU) or mac (Metal/CPU)

pkgname=sd-worker-rbee-bin
pkgver=0.1.0
pkgrel=1
pkgdesc="Stable Diffusion worker for rbee system (pre-built binary)"
arch=('x86_64' 'aarch64')
url="https://github.com/rbee-keeper/rbee"
license=('GPL-3.0-or-later')
provides=('sd-worker-rbee')
conflicts=('sd-worker-rbee-git')

# Auto-detect platform and best GPU backend
_detect_platform() {
    local os=$(uname -s | tr '[:upper:]' '[:lower:]')
    local arch=$(uname -m)
    
    if [[ "$os" == "darwin" ]]; then
        # macOS: Metal (Apple Silicon) or CPU (Intel)
        if [[ "$arch" == "arm64" ]]; then
            echo "macos-aarch64-metal"
        else
            echo "macos-x86_64-cpu"
        fi
    elif [[ "$os" == "linux" ]]; then
        # Linux: CUDA > ROCm > CPU
        if command -v nvidia-smi &> /dev/null; then
            echo "linux-x86_64-cuda"
        elif command -v rocminfo &> /dev/null; then
            echo "linux-x86_64-rocm"
        else
            echo "linux-$arch-cpu"
        fi
    else
        echo "unknown"
    fi
}

_platform=$(_detect_platform)
echo "Detected platform: $_platform"

# Download URL based on detected platform
source=("https://github.com/rbee-keeper/rbee/releases/download/v${pkgver}/sd-worker-rbee-${_platform}-${pkgver}.tar.gz")
sha256sums=('SKIP')

# Dynamic dependencies based on platform
depends=('gcc')
case "$_platform" in
    *cuda*) depends+=('cuda') ;;
    *rocm*) depends+=('rocm') ;;
esac

package() {
    install -Dm755 "$srcdir/sd-worker-rbee" \
        "$pkgdir/usr/local/bin/sd-worker-rbee"
}
