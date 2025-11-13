#!/bin/bash
# Created by: TEAM-488
# Delete ROCm test instance and cleanup

set -e

source .env

echo "=== Cleanup ROCm Test Instance ==="

if [ -f .instance-id ]; then
    INSTANCE_ID=$(cat .instance-id)
    
    echo "Terminating instance $INSTANCE_ID..."
    aws ec2 terminate-instances \
        --instance-ids $INSTANCE_ID \
        --region $AWS_REGION
    
    echo "Waiting for termination..."
    aws ec2 wait instance-terminated \
        --instance-ids $INSTANCE_ID \
        --region $AWS_REGION
    
    rm .instance-id .instance-ip
    echo "✅ Instance terminated"
else
    echo "No instance found"
fi

# Optionally delete security group
read -p "Delete security group? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    aws ec2 delete-security-group \
        --group-name $SECURITY_GROUP \
        --region $AWS_REGION
    echo "✅ Security group deleted"
fi

echo ""
echo "✅ Cleanup complete!"
