# Created by: TEAM-488
# AWS-specific resources for ROCm testing

provider "aws" {
  region = var.region
}

# AMI for Ubuntu 22.04 with ROCm
data "aws_ami" "ubuntu_rocm" {
  count       = var.provider == "aws" ? 1 : 0
  most_recent = true
  owners      = ["099720109477"]  # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Security group
resource "aws_security_group" "rocm_test" {
  count       = var.provider == "aws" ? 1 : 0
  name        = "rbee-rocm-test-sg"
  description = "Security group for rbee ROCm testing"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "rbee-rocm-test-sg"
  })
}

# Key pair
resource "aws_key_pair" "rocm_test" {
  count      = var.provider == "aws" ? 1 : 0
  key_name   = "rbee-rocm-test"
  public_key = file(var.ssh_public_key_path)

  tags = local.common_tags
}

# User data script
data "template_file" "user_data" {
  count    = var.provider == "aws" ? 1 : 0
  template = file("${path.module}/user-data.sh")

  vars = {
    auto_shutdown_hours = var.auto_shutdown_hours
  }
}

# EC2 instance
resource "aws_instance" "rocm_test" {
  count         = var.provider == "aws" ? 1 : 0
  ami           = data.aws_ami.ubuntu_rocm[0].id
  instance_type = var.instance_type
  key_name      = aws_key_pair.rocm_test[0].key_name

  vpc_security_group_ids = [aws_security_group.rocm_test[0].id]

  user_data = data.template_file.user_data[0].rendered

  root_block_device {
    volume_size = 100  # GB
    volume_type = "gp3"
  }

  tags = merge(local.common_tags, {
    Name         = "rbee-rocm-test"
    AutoShutdown = "true"
  })

  lifecycle {
    create_before_destroy = true
  }
}

# CloudWatch alarm for cost monitoring
resource "aws_cloudwatch_metric_alarm" "high_cost" {
  count               = var.provider == "aws" ? 1 : 0
  alarm_name          = "rbee-rocm-test-high-cost"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = "21600"  # 6 hours
  statistic           = "Maximum"
  threshold           = "50"  # $50
  alarm_description   = "Alert when estimated charges exceed $50"

  dimensions = {
    Currency = "USD"
  }
}
