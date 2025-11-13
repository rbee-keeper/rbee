#!/bin/bash
# Created by: TEAM-488
# Run ROCm tests on cloud instance

set -e

if [ ! -f .instance-ip ]; then
    echo "❌ No instance found. Run ./provision.sh first."
    exit 1
fi

PUBLIC_IP=$(cat .instance-ip)

echo "=== Running ROCm Tests on $PUBLIC_IP ==="

ssh -i ~/.ssh/rbee-rocm-test.pem \
    -o StrictHostKeyChecking=no \
    ubuntu@$PUBLIC_IP << 'ENDSSH'
cd ~/rbee

# Setup environment
source $HOME/.cargo/env
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

echo "=== Phase 0: rocm-rs Tests ==="
cd deps/rocm-rs
cargo test
cd ../..

echo ""
echo "=== Phase 1: Candle Device Tests ==="
cd deps/candle/candle-core
cargo test --features rocm rocm_tests
cd ../../..

echo ""
echo "=== Phase 2: Kernel Tests ==="
cd deps/candle/candle-kernels
cargo test --features rocm rocm_kernel_tests
cd ../../..

echo ""
echo "=== Phase 3: Backend Operations Tests ==="
cd deps/candle/candle-core
cargo test --features rocm rocm_ops_tests
cd ../../..

echo ""
echo "✅ All tests complete!"
ENDSSH

echo ""
echo "✅ Tests complete!"
echo ""
echo "Check results above. If all passed, run ./cleanup.sh"
