# rbee: Turn Your GPU Farm Into One Unified API

**Version:** 2.0  
**Date:** November 2, 2025  
**Status:** Complete

---

## Quick Navigation

This document has been split into focused parts for easier reading:

1. **[Executive Summary](01_EXECUTIVE_SUMMARY.md)** - 2-page overview for decision makers
2. **[Consumer Use Case](02_CONSUMER_USE_CASE.md)** - For homelab users and power users
3. **[Business Use Case](03_BUSINESS_USE_CASE.md)** - For GPU infrastructure operators
4. **[Technical Differentiators](04_TECHNICAL_DIFFERENTIATORS.md)** - Why rbee vs alternatives
5. **[Revenue Models](05_REVENUE_MODELS.md)** - Business models and pricing
6. **[Implementation Roadmap](06_IMPLEMENTATION_ROADMAP.md)** - What's ready now vs future

**See [README.md](README.md) for the complete documentation index.**

---

## Overview

**For Consumers:** Stop juggling multiple AI tools. Use ALL your GPUs across ALL your computers with one simple API.

**For Businesses:** Turn your GPU infrastructure into a production-ready AI platform with text, images, video, and audio—all from one endpoint.

---

## The Consumer Story: "I Have Multiple GPUs, Why Can't I Use Them All?"

### The Problem

You have:
- Gaming PC with RTX 4090 (great for Stable Diffusion)
- Mac Studio with M2 Ultra (great for LLMs)
- Old server with 2x RTX 3090 (sitting idle)

**Current reality:**
```bash
# Want to generate an image?
cd ~/stable-diffusion-webui
./webui.sh  # Uses RTX 4090

# Want to chat with an LLM?
cd ~/ollama
ollama run llama3  # Uses... which GPU? Who knows!

# Want to use both at the same time?
# Good luck! They fight over GPU memory
# You're manually switching between programs
# Each has different APIs, different configs
```

### The rbee Solution

```bash
# One-time setup (5 minutes):
rbee hive install gaming-pc    # RTX 4090
rbee hive install mac-studio   # M2 Ultra
rbee hive install old-server   # 2x RTX 3090

# Now use EVERYTHING through one API:
curl http://localhost:7833/v1/chat/completions \
  -d '{"model": "llama-3-70b", "messages": [...]}'
# ↑ Automatically uses Mac M2 Ultra

curl http://localhost:7833/v1/images/generations \
  -d '{"model": "sdxl", "prompt": "a cat"}'
# ↑ Automatically uses RTX 4090

# Both run AT THE SAME TIME
# No conflicts, no manual switching
# One API for everything
```

### The Power User Feature: Pin Models to Specific GPUs

**Option 1: GUI (Point and Click)**
```
Open http://localhost:7833/ui

Hive: gaming-pc (RTX 4090)
  ├─ Worker 1: SDXL (pinned) ✓
  └─ Worker 2: Available

Hive: mac-studio (M2 Ultra)  
  ├─ Worker 1: llama-3-70b (pinned) ✓
  └─ Worker 2: Available

Hive: old-server (2x RTX 3090)
  ├─ Worker 1: Available
  └─ Worker 2: Available
```

**Option 2: Rhai Script (Programmable)**
```rhai
// ~/.config/rbee/scheduler.rhai

fn route_task(task, workers) {
    // Images always go to RTX 4090
    if task.type == "image-gen" {
        return workers
            .filter(|w| w.hive == "gaming-pc")
            .first();
    }
    
    // Large LLMs go to Mac M2 Ultra
    if task.type == "text-gen" && task.model.contains("70b") {
        return workers
            .filter(|w| w.hive == "mac-studio")
            .first();
    }
    
    // Everything else: least loaded GPU
    return workers.least_loaded();
}
```

**Result:** You control exactly which GPU does what, without touching code or restarting services.

---

## The Business Story: "Turn Your GPU Farm Into a Product"

### The Problem

You're a business with GPU infrastructure:
- 20x NVIDIA A100 GPUs
- 10x H100 GPUs  
- Mix of different models and capabilities

**You want to offer AI services to customers, but:**
- ❌ Setting up text + images + audio = 3 different platforms
- ❌ Each platform has different APIs, configs, monitoring
- ❌ Customers need different API keys for each service
- ❌ You can't easily control which customer uses which GPU
- ❌ No built-in GDPR compliance
- ❌ Scaling means managing 3+ separate systems

### The rbee Solution

**One platform. One API. All modalities.**

```bash
# Setup (one time):
rbee hive install gpu-node-01  # 4x A100
rbee hive install gpu-node-02  # 4x A100
rbee hive install gpu-node-03  # 2x H100
# ... repeat for all nodes

# Configure your model catalog:
rbee model add llama-3-405b --hive gpu-node-03  # H100 for large models
rbee model add sdxl --hive gpu-node-01          # A100 for images
rbee model add whisper-large --hive gpu-node-02 # A100 for audio

# Set up scheduler (Rhai script):
cat > ~/.config/rbee/scheduler.rhai << 'EOF'
fn route_task(task, workers) {
    // Enterprise customers get H100s
    if task.customer_tier == "enterprise" {
        return workers.filter(|w| w.gpu_type == "H100").least_loaded();
    }
    
    // Free tier gets A100s only
    if task.customer_tier == "free" {
        return workers.filter(|w| w.gpu_type == "A100").least_loaded();
    }
    
    // Route by task type
    match task.type {
        "image-gen" => workers.filter(|w| w.capability == "image-gen").least_loaded(),
        "audio-gen" => workers.filter(|w| w.capability == "audio-gen").least_loaded(),
        _ => workers.least_loaded()
    }
}
EOF

# Your customers get ONE API endpoint:
# https://api.yourcompany.com/v1/
```

### What Your Customers See

**One OpenAI-compatible API for everything:**

```bash
# Text generation
curl https://api.yourcompany.com/v1/chat/completions \
  -H "Authorization: Bearer customer-key-123" \
  -d '{"model": "llama-3-405b", "messages": [...]}'

# Image generation  
curl https://api.yourcompany.com/v1/images/generations \
  -H "Authorization: Bearer customer-key-123" \
  -d '{"model": "sdxl", "prompt": "a cat"}'

# Audio transcription
curl https://api.yourcompany.com/v1/audio/transcriptions \
  -H "Authorization: Bearer customer-key-123" \
  -F file=@audio.mp3

# All from the SAME endpoint
# All with the SAME API key
# All billed together
```

### Business Features Out of the Box

**1. Multi-Tenancy & Quotas**
```rhai
fn should_admit(task, customer) {
    // Check quota
    if customer.tokens_used_today > customer.daily_limit {
        return reject("Quota exceeded");
    }
    
    // Check tier access
    if task.model == "llama-3-405b" && customer.tier != "enterprise" {
        return reject("Upgrade to enterprise");
    }
    
    return admit();
}
```

**2. GDPR Compliance (Built-in)**
- ✅ Immutable audit logs (7-year retention)
- ✅ Data export endpoints (`/gdpr/export`)
- ✅ Data deletion endpoints (`/gdpr/delete`)
- ✅ EU-only worker filtering
- ✅ Consent tracking

**3. Cost Control**
```rhai
fn route_task(task, workers) {
    // Route cheap models to cheap GPUs
    if task.model == "llama-3-8b" {
        return workers.filter(|w| w.gpu_type == "RTX4090").first();
    }
    
    // Route expensive models to expensive GPUs only when needed
    if task.model == "llama-3-405b" {
        return workers.filter(|w| w.gpu_type == "H100").first();
    }
}
```

**4. Custom Model Catalog**
```bash
# Offer your fine-tuned models
rbee model add your-custom-llm-v2 --hive gpu-node-01
rbee model add your-custom-sdxl --hive gpu-node-02

# Customers access them via API:
curl https://api.yourcompany.com/v1/models
# {
#   "models": [
#     {"id": "your-custom-llm-v2", "type": "text"},
#     {"id": "your-custom-sdxl", "type": "image"}
#   ]
# }
```

---

## Comparison: Consumer vs Business

| Feature | Consumer (Homelab) | Business (GPU Farm) |
|---------|-------------------|---------------------|
| **Setup** | `rbee hive install` for each computer | `rbee hive install` for each node |
| **Scheduler** | GUI or custom Rhai script | Recommended Rhai script (multi-tenant) |
| **Security** | Local network only | TLS + API keys + audit logs |
| **Models** | Download from HuggingFace | Custom fine-tuned models |
| **Monitoring** | Web UI dashboard | Web UI + Prometheus metrics |
| **Cost** | Free (electricity only) | Platform fee or self-hosted |

---

## The Value Propositions

### For Consumers

**"Stop juggling AI tools. One API for everything."**

- ✅ Use ALL your GPUs (Mac, Windows, Linux)
- ✅ Mix CUDA, Metal, CPU freely
- ✅ Pin models to specific GPUs (GUI or script)
- ✅ OpenAI-compatible (works with Zed, Cursor, etc.)
- ✅ No cloud costs

### For Businesses

**"Turn your GPU farm into a production AI platform in one day."**

- ✅ One API endpoint for text, images, audio, video
- ✅ Multi-tenancy out of the box (Rhai scheduler)
- ✅ GDPR compliance built-in (EU market ready)
- ✅ Custom model catalog (your fine-tuned models)
- ✅ Cost control (route by customer tier, model size)
- ✅ 50-70% cheaper than cloud (consumer GPUs)

---

## Technical Differentiators

### vs ComfyUI + Ollama + Whisper (Consumer)

**Current setup:**
- ComfyUI for images (port 8188)
- Ollama for text (port 11434)
- Whisper for audio (separate Python script)
- Each fights for GPU memory
- Different APIs, different configs

**rbee:**
- One API for everything (port 7833)
- Automatic GPU allocation
- No conflicts, no manual switching
- OpenAI-compatible

### vs Building Your Own Platform (Business)

**Building from scratch:**
- 6-12 months development
- vLLM + ComfyUI + Whisper integration
- Custom load balancer
- Custom billing system
- Custom monitoring
- GDPR compliance from scratch

**rbee:**
- Setup in 1 day
- All integrations included
- Rhai scheduler (no code changes)
- Audit logging built-in
- GDPR endpoints included
- Web UI included

---

## Revenue Models

### Consumer: Open Source (GPL-3.0)
- Free forever
- Community support
- Self-hosted only

### Business: Two Options

**Option 1: Self-Hosted (GPL-3.0)**
- Free software
- You run it on your infrastructure
- You keep 100% of revenue
- Community support

**Option 2: Managed Platform (Future)**
- We run the infrastructure
- You get API endpoint
- 30-40% platform fee
- Enterprise support
- SLA guarantees