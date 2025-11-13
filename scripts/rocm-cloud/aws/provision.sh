#!/bin/bash
# Created by: TEAM-488
# Provision AWS EC2 instance with AMD GPU for ROCm testing

set -e

# Load configuration
source .env

echo "=== Provisioning AMD GPU Instance ==="
echo "Region: $AWS_REGION"
echo "Instance Type: $INSTANCE_TYPE"
echo "Max Hours: $MAX_HOURS"
echo ""

# Create security group if it doesn't exist
if ! aws ec2 describe-security-groups --group-names $SECURITY_GROUP &> /dev/null; then
    echo "Creating security group..."
    aws ec2 create-security-group \
        --group-name $SECURITY_GROUP \
        --description "Security group for rbee ROCm testing" \
        --region $AWS_REGION
    
    # Allow SSH
    aws ec2 authorize-security-group-ingress \
        --group-name $SECURITY_GROUP \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0 \
        --region $AWS_REGION
    
    echo "✅ Security group created"
fi

# User data script (runs on instance startup)
cat > user-data.sh << 'USERDATA'
#!/bin/bash
# Instance initialization script

# Update system
apt-get update
apt-get upgrade -y

# Install ROCm
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb
apt-get install -y ./amdgpu-install_6.0.60000-1_all.deb
amdgpu-install --usecase=rocm --no-dkms -y

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# Install build dependencies
apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libssl-dev \
    clang \
    llvm

# Setup auto-shutdown (4 hours)
cat > /usr/local/bin/auto-shutdown.sh << 'EOF'
#!/bin/bash
UPTIME_HOURS=$(awk '{print int($1/3600)}' /proc/uptime)
if [ $UPTIME_HOURS -ge 4 ]; then
    echo "Instance running for $UPTIME_HOURS hours. Shutting down..."
    shutdown -h now
fi
EOF
chmod +x /usr/local/bin/auto-shutdown.sh

# Add to cron (check every 30 minutes)
echo "*/30 * * * * /usr/local/bin/auto-shutdown.sh" | crontab -

# Create marker file
touch /var/lib/cloud/instance-ready

echo "Instance setup complete!"
USERDATA

# Launch instance
echo "Launching instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-groups $SECURITY_GROUP \
    --user-data file://user-data.sh \
    --tag-specifications "ResourceType=instance,Tags=[
        {Key=Name,Value=rbee-rocm-test},
        {Key=Project,Value=$PROJECT},
        {Key=Component,Value=$COMPONENT},
        {Key=Team,Value=$TEAM},
        {Key=AutoShutdown,Value=true}
    ]" \
    --region $AWS_REGION \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "✅ Instance launched: $INSTANCE_ID"
echo "$INSTANCE_ID" > .instance-id

# Wait for instance to be running
echo "Waiting for instance to start..."
aws ec2 wait instance-running \
    --instance-ids $INSTANCE_ID \
    --region $AWS_REGION

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $AWS_REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "$PUBLIC_IP" > .instance-ip

echo ""
echo "✅ Instance ready!"
echo ""
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo ""
echo "Waiting 2 minutes for initialization to complete..."
sleep 120

echo ""
echo "Next steps:"
echo "  ./ssh.sh          - SSH into instance"
echo "  ./deploy.sh       - Deploy rbee code"
echo "  ./run-tests.sh    - Run ROCm tests"
echo ""
echo "⚠️  IMPORTANT: Run ./cleanup.sh when done to avoid charges!"
echo ""

# Cleanup temp files
rm user-data.sh
