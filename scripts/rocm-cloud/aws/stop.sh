#!/bin/bash
# Created by: TEAM-488
# Stop (but don't delete) ROCm test instance

set -e

source .env

if [ ! -f .instance-id ]; then
    echo "❌ No instance found."
    exit 1
fi

INSTANCE_ID=$(cat .instance-id)

echo "Stopping instance $INSTANCE_ID..."
aws ec2 stop-instances \
    --instance-ids $INSTANCE_ID \
    --region $AWS_REGION

echo "✅ Instance stopped (not deleted)"
echo ""
echo "To restart: aws ec2 start-instances --instance-ids $INSTANCE_ID"
echo "To delete: ./cleanup.sh"
