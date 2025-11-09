#!/bin/bash
# rbee-keeper installation script
# Usage: curl -fsSL https://install.rbee.dev | sh
# Or: curl -fsSL https://install.rbee.dev | sh -s -- --dev

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VERSION="${RBEE_VERSION:-0.1.0}"
BUILD_TYPE="prod"  # prod or dev
INSTALL_DIR="${RBEE_INSTALL_DIR:-$HOME/.local/bin}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            BUILD_TYPE="dev"
            shift
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Detect OS and architecture
detect_platform() {
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
            echo -e "${RED}Unsupported OS: $OS${NC}"
            exit 1
            ;;
    esac
    
    case "$ARCH" in
        x86_64|amd64)
            ARCH="x86_64"
            ;;
        aarch64|arm64)
            ARCH="arm64"
            ;;
        *)
            echo -e "${RED}Unsupported architecture: $ARCH${NC}"
            exit 1
            ;;
    esac
}

# Download and install rbee-keeper
install_rbee_keeper() {
    echo -e "${BLUE}🐝 Installing rbee-keeper...${NC}"
    echo -e "${BLUE}   Platform: $PLATFORM-$ARCH${NC}"
    echo -e "${BLUE}   Build: $BUILD_TYPE${NC}"
    echo -e "${BLUE}   Version: $VERSION${NC}"
    echo ""
    
    if [ "$BUILD_TYPE" = "prod" ]; then
        # Download pre-built binary from GitHub releases
        BINARY_URL="https://github.com/rbee-keeper/rbee/releases/download/v${VERSION}/rbee-keeper-${PLATFORM}-${ARCH}-${VERSION}.tar.gz"
        
        echo -e "${YELLOW}📥 Downloading from GitHub releases...${NC}"
        echo -e "${BLUE}   URL: $BINARY_URL${NC}"
        
        TMP_DIR="$(mktemp -d)"
        cd "$TMP_DIR"
        
        if ! curl -fsSL "$BINARY_URL" -o rbee-keeper.tar.gz; then
            echo -e "${RED}❌ Failed to download rbee-keeper${NC}"
            echo -e "${YELLOW}💡 Try development build: curl -fsSL https://install.rbee.dev | sh -s -- --dev${NC}"
            exit 1
        fi
        
        echo -e "${GREEN}✅ Downloaded${NC}"
        echo -e "${YELLOW}📦 Extracting...${NC}"
        
        tar -xzf rbee-keeper.tar.gz
        
    else
        # Build from source
        echo -e "${YELLOW}🔨 Building from source (this may take a while)...${NC}"
        
        if ! command -v cargo &> /dev/null; then
            echo -e "${RED}❌ Rust/Cargo not found${NC}"
            echo -e "${YELLOW}💡 Install Rust: https://rustup.rs${NC}"
            exit 1
        fi
        
        if ! command -v git &> /dev/null; then
            echo -e "${RED}❌ Git not found${NC}"
            exit 1
        fi
        
        TMP_DIR="$(mktemp -d)"
        cd "$TMP_DIR"
        
        echo -e "${YELLOW}📥 Cloning repository...${NC}"
        git clone --depth 1 --branch development https://github.com/rbee-keeper/rbee.git
        cd rbee
        
        echo -e "${YELLOW}🔨 Building rbee-keeper...${NC}"
        cargo build --release --package rbee-keeper
        
        cp target/release/rbee-keeper "$TMP_DIR/"
        cd "$TMP_DIR"
    fi
    
    # Install binary
    echo -e "${YELLOW}📦 Installing to $INSTALL_DIR...${NC}"
    mkdir -p "$INSTALL_DIR"
    
    if [ -f "$INSTALL_DIR/rbee" ]; then
        echo -e "${YELLOW}⚠️  Existing rbee installation found, backing up...${NC}"
        mv "$INSTALL_DIR/rbee" "$INSTALL_DIR/rbee.backup"
    fi
    
    mv rbee-keeper "$INSTALL_DIR/rbee"
    chmod +x "$INSTALL_DIR/rbee"
    
    # Cleanup
    cd /
    rm -rf "$TMP_DIR"
    
    echo -e "${GREEN}✅ rbee-keeper installed successfully!${NC}"
}

# Check if install directory is in PATH
check_path() {
    if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
        echo ""
        echo -e "${YELLOW}⚠️  $INSTALL_DIR is not in your PATH${NC}"
        echo -e "${YELLOW}   Add this to your shell profile (~/.bashrc, ~/.zshrc, etc.):${NC}"
        echo ""
        echo -e "${BLUE}   export PATH=\"\$PATH:$INSTALL_DIR\"${NC}"
        echo ""
    fi
}

# Print next steps
print_next_steps() {
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}🎉 Installation complete!${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${BLUE}📚 Next steps:${NC}"
    echo ""
    echo -e "${YELLOW}1. Verify installation:${NC}"
    echo -e "   ${BLUE}rbee --version${NC}"
    echo ""
    echo -e "${YELLOW}2. Start queen (orchestrator):${NC}"
    echo -e "   ${BLUE}rbee queen start${NC}"
    echo ""
    echo -e "${YELLOW}3. Create a hive (worker pool):${NC}"
    echo -e "   ${BLUE}rbee hive create my-hive${NC}"
    echo ""
    echo -e "${YELLOW}4. Install a worker:${NC}"
    echo -e "   ${BLUE}rbee worker install llm-worker-rbee-cpu${NC}"
    echo ""
    echo -e "${YELLOW}5. Download a model:${NC}"
    echo -e "   ${BLUE}rbee model download llama-3.2-1b${NC}"
    echo ""
    echo -e "${BLUE}📖 Documentation: ${NC}https://docs.rbee.dev"
    echo -e "${BLUE}💬 Community: ${NC}https://github.com/rbee-keeper/rbee/discussions"
    echo ""
}

# Main installation flow
main() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}🐝 rbee-keeper installer${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    detect_platform
    install_rbee_keeper
    check_path
    print_next_steps
}

main
