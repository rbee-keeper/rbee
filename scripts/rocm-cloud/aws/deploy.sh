#!/bin/bash
# Created by: TEAM-488
# Deploy rbee code to ROCm test instance

set -e

if [ ! -f .instance-ip ]; then
    echo "❌ No instance found. Run ./provision.sh first."
    exit 1
fi

PUBLIC_IP=$(cat .instance-ip)

echo "=== Deploying rbee to $PUBLIC_IP ==="

# Create deployment package
echo "Creating deployment package..."
cd ../../../../
tar czf /tmp/rbee-rocm-test.tar.gz \
    --exclude='target' \
    --exclude='.git' \
    --exclude='node_modules' \
    --exclude='*.log' \
    deps/ bin/ scripts/ Cargo.toml Cargo.lock

# Copy to instance
echo "Uploading code..."
scp -i ~/.ssh/rbee-rocm-test.pem \
    -o StrictHostKeyChecking=no \
    /tmp/rbee-rocm-test.tar.gz \
    ubuntu@$PUBLIC_IP:~/

# Extract and setup
echo "Setting up on instance..."
ssh -i ~/.ssh/rbee-rocm-test.pem \
    -o StrictHostKeyChecking=no \
    ubuntu@$PUBLIC_IP << 'ENDSSH'
# Extract
mkdir -p ~/rbee
cd ~/rbee
tar xzf ~/rbee-rocm-test.tar.gz
rm ~/rbee-rocm-test.tar.gz

# Setup environment
source $HOME/.cargo/env
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# Build rocm-rs
cd deps/rocm-rs
cargo build
cd ../..

# Verify ROCm
echo ""
echo "=== ROCm Verification ==="
rocm-smi
hipcc --version

echo ""
echo "✅ Deployment complete!"
ENDSSH

# Cleanup
rm /tmp/rbee-rocm-test.tar.gz

echo ""
echo "✅ Code deployed!"
echo ""
echo "Next: ./run-tests.sh"
