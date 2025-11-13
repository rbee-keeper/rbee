# ROCm Cloud Testing Infrastructure

**Created by:** TEAM-488  
**Purpose:** Test ROCm integration on cloud AMD GPUs

---

## Overview

Since we don't have local AMD GPUs, this infrastructure provisions cloud instances with AMD GPUs for testing.

**Supported Providers:**
- AWS EC2 (AMD Instinct MI series)
- Azure (AMD GPU VMs)
- Paperspace (AMD instances)

**Estimated Costs:**
- AWS: ~$3-8/hour
- Azure: ~$2-6/hour
- Paperspace: ~$1-3/hour

**Total for 1 week testing:** $50-200

---

## Quick Start

### 1. AWS (Recommended)

```bash
# Setup
cd scripts/rocm-cloud/aws
./setup.sh

# Provision instance
./provision.sh

# SSH into instance
./ssh.sh

# Run tests
./run-tests.sh

# Cleanup (IMPORTANT!)
./cleanup.sh
```

### 2. Terraform (All Providers)

```bash
cd scripts/rocm-cloud/terraform

# AWS
terraform init
terraform plan -var="provider=aws"
terraform apply -var="provider=aws"

# Cleanup
terraform destroy
```

---

## Cost Management

### Auto-Shutdown
All instances have:
- 4-hour idle timeout
- Daily cost alerts
- Auto-shutdown at 2am UTC

### Manual Shutdown
```bash
# AWS
./aws/stop.sh

# Azure
./azure/stop.sh
```

---

## Testing Workflow

### Phase 0-3 (No GPU Needed)
- Develop locally
- Write code
- Unit tests (CPU)

### Phase 4-6 (GPU Required)
1. Provision cloud instance
2. Deploy code
3. Run GPU tests
4. Verify results
5. **Shutdown instance**

---

## Files

```
scripts/rocm-cloud/
├── README.md                 ← This file
├── aws/
│   ├── setup.sh             ← Install AWS CLI
│   ├── provision.sh         ← Create EC2 instance
│   ├── ssh.sh               ← SSH into instance
│   ├── deploy.sh            ← Deploy rbee code
│   ├── run-tests.sh         ← Run ROCm tests
│   ├── stop.sh              ← Stop instance
│   └── cleanup.sh           ← Delete everything
├── azure/
│   ├── setup.sh
│   ├── provision.sh
│   └── ...
├── terraform/
│   ├── main.tf              ← Multi-cloud Terraform
│   ├── aws.tf
│   ├── azure.tf
│   └── variables.tf
└── docker/
    ├── Dockerfile.rocm      ← ROCm test container
    └── docker-compose.yml
```

---

## Security

- SSH keys auto-generated
- Security groups restrict access
- Credentials in `.env` (gitignored)
- Auto-cleanup on exit

---

## Next Steps

1. Choose provider (AWS recommended)
2. Run setup script
3. Provision when ready to test
4. **Remember to cleanup!**
