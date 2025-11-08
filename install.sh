#!/bin/bash
# rbee Universal Installer
# Usage: curl -fsSL https://install.rbee.dev | sh
# Or: curl -fsSL https://raw.githubusercontent.com/rbee-keeper/rbee/production/install.sh | sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VERSION="${VERSION:-latest}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"
GITHUB_REPO="rbee-keeper/rbee"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}🐝 rbee Installer${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Detect platform
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Linux*)
        PLATFORM="linux"
        ;;
    Darwin*)
        PLATFORM="macos"
        ;;
    *)
        echo -e "${RED}❌ Unsupported OS: $OS${NC}"
        echo "rbee currently supports Linux and macOS"
        exit 1
        ;;
esac

# Normalize architecture
case "$ARCH" in
    x86_64|amd64)
        ARCH="x86_64"
        ;;
    aarch64|arm64)
        ARCH="aarch64"
        ;;
    *)
        echo -e "${RED}❌ Unsupported architecture: $ARCH${NC}"
        exit 1
        ;;
esac

echo -e "Platform: ${GREEN}$PLATFORM-$ARCH${NC}"
echo -e "Version: ${GREEN}$VERSION${NC}"
echo -e "Install directory: ${GREEN}$INSTALL_DIR${NC}"
echo ""

# Get latest version if not specified
if [ "$VERSION" = "latest" ]; then
    echo "🔍 Fetching latest version..."
    VERSION=$(curl -fsSL "https://api.github.com/repos/$GITHUB_REPO/releases/latest" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')
    VERSION="${VERSION#v}"  # Remove 'v' prefix
    echo -e "Latest version: ${GREEN}$VERSION${NC}"
    echo ""
fi

# Construct download URL
TARBALL="rbee-$PLATFORM-$ARCH.tar.gz"
DOWNLOAD_URL="https://github.com/$GITHUB_REPO/releases/download/v$VERSION/$TARBALL"

echo "📥 Downloading rbee..."
echo -e "${BLUE}$DOWNLOAD_URL${NC}"
echo ""

# Create temp directory
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

# Download and extract
if ! curl -fsSL "$DOWNLOAD_URL" -o "$TMP_DIR/$TARBALL"; then
    echo -e "${RED}❌ Download failed${NC}"
    echo ""
    echo "Possible reasons:"
    echo "  - Version $VERSION doesn't exist"
    echo "  - No build for $PLATFORM-$ARCH"
    echo "  - Network error"
    echo ""
    echo "Available releases:"
    echo "  https://github.com/$GITHUB_REPO/releases"
    exit 1
fi

echo "📦 Extracting..."
tar -xzf "$TMP_DIR/$TARBALL" -C "$TMP_DIR"

# Create install directory
mkdir -p "$INSTALL_DIR"

# Install binaries
echo "📥 Installing binaries to $INSTALL_DIR..."

BINARIES=("rbee-keeper" "queen-rbee" "rbee-hive")
for binary in "${BINARIES[@]}"; do
    if [ -f "$TMP_DIR/$binary" ]; then
        cp "$TMP_DIR/$binary" "$INSTALL_DIR/"
        chmod +x "$INSTALL_DIR/$binary"
        echo -e "  ${GREEN}✓${NC} $binary"
    else
        echo -e "  ${YELLOW}⚠${NC} $binary not found (optional)"
    fi
done

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ rbee installed successfully!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check if install dir is in PATH
if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
    echo -e "${YELLOW}⚠️  $INSTALL_DIR is not in your PATH${NC}"
    echo ""
    echo "Add to your shell profile:"
    echo ""
    
    # Detect shell
    SHELL_NAME=$(basename "$SHELL")
    case "$SHELL_NAME" in
        bash)
            PROFILE="$HOME/.bashrc"
            ;;
        zsh)
            PROFILE="$HOME/.zshrc"
            ;;
        fish)
            PROFILE="$HOME/.config/fish/config.fish"
            ;;
        *)
            PROFILE="$HOME/.profile"
            ;;
    esac
    
    echo -e "  ${BLUE}echo 'export PATH=\"$INSTALL_DIR:\$PATH\"' >> $PROFILE${NC}"
    echo -e "  ${BLUE}source $PROFILE${NC}"
    echo ""
fi

echo "🚀 Get started:"
echo ""
echo -e "  ${BLUE}rbee-keeper --help${NC}    # CLI tool"
echo -e "  ${BLUE}queen-rbee --help${NC}     # Orchestrator daemon"
echo -e "  ${BLUE}rbee-hive --help${NC}      # Pool manager"
echo ""
echo "📚 Documentation:"
echo "  https://github.com/$GITHUB_REPO"
echo ""
