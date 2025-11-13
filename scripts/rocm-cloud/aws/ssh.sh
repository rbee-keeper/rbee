#!/bin/bash
# Created by: TEAM-488
# SSH into ROCm test instance

set -e

if [ ! -f .instance-ip ]; then
    echo "❌ No instance found. Run ./provision.sh first."
    exit 1
fi

PUBLIC_IP=$(cat .instance-ip)

echo "Connecting to $PUBLIC_IP..."
ssh -i ~/.ssh/rbee-rocm-test.pem \
    -o StrictHostKeyChecking=no \
    ubuntu@$PUBLIC_IP
