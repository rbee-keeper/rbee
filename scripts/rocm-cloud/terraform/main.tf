# Created by: TEAM-488
# Terraform configuration for ROCm cloud testing
# Supports AWS, Azure, and GCP

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

# Variables
variable "provider" {
  description = "Cloud provider (aws, azure)"
  type        = string
  default     = "aws"
}

variable "region" {
  description = "Cloud region"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "Instance type with AMD GPU"
  type        = string
  default     = "g4ad.xlarge"  # AWS default
}

variable "ssh_public_key_path" {
  description = "Path to SSH public key"
  type        = string
  default     = "~/.ssh/id_rsa.pub"
}

variable "auto_shutdown_hours" {
  description = "Auto-shutdown after N hours"
  type        = number
  default     = 4
}

# Locals
locals {
  common_tags = {
    Project   = "rbee"
    Component = "rocm-testing"
    Team      = "TEAM-488"
    ManagedBy = "Terraform"
  }
}

# Outputs
output "instance_id" {
  description = "Instance ID"
  value       = var.provider == "aws" ? aws_instance.rocm_test[0].id : null
}

output "public_ip" {
  description = "Public IP address"
  value       = var.provider == "aws" ? aws_instance.rocm_test[0].public_ip : null
}

output "ssh_command" {
  description = "SSH command to connect"
  value       = var.provider == "aws" ? "ssh -i ~/.ssh/rbee-rocm-test.pem ubuntu@${aws_instance.rocm_test[0].public_ip}" : null
}
