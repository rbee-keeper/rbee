#!/bin/bash
# Created by: TEAM-488
# Setup AWS CLI and credentials for ROCm testing

set -e

echo "=== AWS ROCm Testing Setup ==="

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "Installing AWS CLI..."
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
    rm -rf aws awscliv2.zip
fi

echo "✅ AWS CLI installed: $(aws --version)"

# Configure AWS credentials
if [ ! -f ~/.aws/credentials ]; then
    echo ""
    echo "AWS credentials not found. Let's configure them."
    echo "You'll need:"
    echo "  - AWS Access Key ID"
    echo "  - AWS Secret Access Key"
    echo "  - Default region (us-east-1 recommended for AMD GPUs)"
    echo ""
    aws configure
else
    echo "✅ AWS credentials already configured"
fi

# Create SSH key if needed
if [ ! -f ~/.ssh/rbee-rocm-test.pem ]; then
    echo "Creating SSH key pair..."
    aws ec2 create-key-pair \
        --key-name rbee-rocm-test \
        --query 'KeyMaterial' \
        --output text > ~/.ssh/rbee-rocm-test.pem
    chmod 400 ~/.ssh/rbee-rocm-test.pem
    echo "✅ SSH key created: ~/.ssh/rbee-rocm-test.pem"
else
    echo "✅ SSH key already exists"
fi

# Create .env file
cat > .env << 'EOF'
# AWS Configuration for ROCm Testing
# Created by: TEAM-488

# Instance Configuration
AWS_REGION=us-east-1
INSTANCE_TYPE=g4ad.xlarge  # AMD GPU instance
AMI_ID=ami-0c7217cdde317cfec  # Ubuntu 22.04 with ROCm
KEY_NAME=rbee-rocm-test
SECURITY_GROUP=rbee-rocm-test-sg

# Cost Management
MAX_HOURS=4  # Auto-shutdown after 4 hours
ALERT_EMAIL=your-email@example.com

# Tags
PROJECT=rbee
COMPONENT=rocm-testing
TEAM=TEAM-488
EOF

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env and set your ALERT_EMAIL"
echo "  2. Run ./provision.sh to create instance"
echo ""
