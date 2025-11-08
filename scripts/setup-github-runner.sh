#!/bin/bash
# Setup GitHub Actions self-hosted runner on macOS
# Created by: TEAM-451
# Usage: ./setup-github-runner.sh YOUR_GITHUB_TOKEN

set -e

TOKEN="$1"

if [[ -z "$TOKEN" ]]; then
    echo "❌ Error: GitHub token required"
    echo ""
    echo "Usage: $0 YOUR_GITHUB_TOKEN"
    echo ""
    echo "Get your token from:"
    echo "https://github.com/rbee-keeper/rbee/settings/actions/runners/new"
    echo ""
    echo "Steps:"
    echo "1. Go to the URL above"
    echo "2. Select: macOS + ARM64"
    echo "3. Copy the token from the './config.sh --token XXXXX' command"
    echo "4. Run: $0 XXXXX"
    exit 1
fi

echo "🤖 Setting up GitHub Actions Runner on macOS..."
echo ""

# Check if running on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "❌ This script must run on macOS"
    exit 1
fi

# Create actions-runner directory
echo "📁 Creating actions-runner directory..."
mkdir -p ~/actions-runner
cd ~/actions-runner

# Download latest runner
echo "📥 Downloading GitHub Actions Runner..."
RUNNER_VERSION="2.321.0"
curl -o actions-runner-osx-arm64-${RUNNER_VERSION}.tar.gz -L \
  https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-osx-arm64-${RUNNER_VERSION}.tar.gz

# Extract
echo "📦 Extracting..."
tar xzf ./actions-runner-osx-arm64-${RUNNER_VERSION}.tar.gz
rm actions-runner-osx-arm64-${RUNNER_VERSION}.tar.gz

# Configure
echo "⚙️  Configuring runner..."
./config.sh \
  --url https://github.com/rbee-keeper/rbee \
  --token "$TOKEN" \
  --name "mac" \
  --labels "macos,arm64,self-hosted" \
  --work _work \
  --replace

# Install as service
echo "🔧 Installing as service..."
./svc.sh install

# Start service
echo "🚀 Starting service..."
./svc.sh start

# Check status
echo ""
echo "✅ GitHub Actions Runner installed and started!"
echo ""
echo "Status:"
./svc.sh status

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Setup complete!"
echo ""
echo "Verify on GitHub:"
echo "  https://github.com/rbee-keeper/rbee/settings/actions/runners"
echo ""
echo "Watch logs:"
echo "  tail -f ~/actions-runner/_diag/Runner_*.log"
echo ""
echo "Control service:"
echo "  cd ~/actions-runner"
echo "  ./svc.sh status   # Check status"
echo "  ./svc.sh stop     # Stop runner"
echo "  ./svc.sh start    # Start runner"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
