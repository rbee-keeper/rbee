#!/bin/bash
# Created by: TEAM-488
# User data script for ROCm instance initialization

set -e

# Update system
apt-get update
apt-get upgrade -y

# Install ROCm
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb
apt-get install -y ./amdgpu-install_6.0.60000-1_all.deb
amdgpu-install --usecase=rocm --no-dkms -y

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Install build dependencies
apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libssl-dev \
    clang \
    llvm \
    python3-pip

# Setup auto-shutdown
cat > /usr/local/bin/auto-shutdown.sh << 'EOF'
#!/bin/bash
UPTIME_HOURS=$(awk '{print int($1/3600)}' /proc/uptime)
if [ $UPTIME_HOURS -ge ${auto_shutdown_hours} ]; then
    echo "Instance running for $UPTIME_HOURS hours. Shutting down..."
    shutdown -h now
fi
EOF
chmod +x /usr/local/bin/auto-shutdown.sh

# Add to cron
echo "*/30 * * * * /usr/local/bin/auto-shutdown.sh" | crontab -

# Mark as ready
touch /var/lib/cloud/instance-ready

echo "ROCm instance initialization complete!"
