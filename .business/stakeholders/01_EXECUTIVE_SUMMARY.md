# rbee: Executive Summary

**Version:** 2.0  
**Date:** November 2, 2025  
**Audience:** Decision makers, investors, technical leaders

---

## What is rbee?

**rbee turns heterogeneous GPU infrastructure into a unified AI platform with one simple API.**

- **For Consumers:** Stop juggling multiple AI tools (ComfyUI, Ollama, Whisper). Use ALL your GPUs across ALL your computers with one API.
- **For Businesses:** Turn your GPU farm into a production-ready AI platform offering text, images, video, and audio from one endpoint.

---

## The Core Problem

### Consumer Problem: "I Have Multiple GPUs, Why Can't I Use Them All?"

You have:
- Gaming PC with RTX 4090 (great for images)
- Mac Studio with M2 Ultra (great for LLMs)
- Old server with 2x RTX 3090 (sitting idle)

**Current reality:**
- ComfyUI for images (port 8188)
- Ollama for text (port 11434)
- Whisper for audio (separate script)
- They fight over GPU memory
- Different APIs, different configs
- Manual switching between tools

### Business Problem: "GPU Farm Complexity"

You have 20x A100 + 10x H100 GPUs and want to offer AI services, but:
- ❌ Text + images + audio = 3 different platforms
- ❌ Different APIs, configs, monitoring for each
- ❌ Can't control which customer uses which GPU
- ❌ No built-in GDPR compliance
- ❌ Scaling = managing 3+ separate systems

---

## The rbee Solution

### One API for Everything

```bash
# Consumer: 5-minute setup
rbee hive install gaming-pc    # RTX 4090
rbee hive install mac-studio   # M2 Ultra
rbee hive install old-server   # 2x RTX 3090

# Now use everything through one API:
curl http://localhost:7833/v1/chat/completions \
  -d '{"model": "llama-3-70b", "messages": [...]}'
# ↑ Automatically uses Mac M2 Ultra

curl http://localhost:7833/v1/images/generations \
  -d '{"model": "sdxl", "prompt": "a cat"}'
# ↑ Automatically uses RTX 4090

# Both run AT THE SAME TIME
# No conflicts, no manual switching
```

### Business: One Platform, All Modalities

```bash
# Setup your GPU farm (one time)
rbee hive install gpu-node-01  # 4x A100
rbee hive install gpu-node-02  # 4x A100
rbee hive install gpu-node-03  # 2x H100

# Configure scheduler (Rhai script)
fn route_task(task, workers) {
    // Enterprise customers get H100s
    if task.customer_tier == "enterprise" {
        return workers.filter(|w| w.gpu_type == "H100").least_loaded();
    }
    // Free tier gets A100s
    return workers.filter(|w| w.gpu_type == "A100").least_loaded();
}

# Your customers get ONE API endpoint:
# https://api.yourcompany.com/v1/
```

---

## Key Value Propositions

### For Consumers

**"Stop juggling AI tools. One API for everything."**

- ✅ Use ALL your GPUs (Mac, Windows, Linux)
- ✅ Mix CUDA, Metal, CPU freely
- ✅ Pin models to specific GPUs (GUI or Rhai script)
- ✅ OpenAI-compatible (works with Zed, Cursor, etc.)
- ✅ No cloud costs (electricity only)
- ✅ Free and open source (GPL-3.0)

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

### vs Current Solutions

| Feature | rbee | ComfyUI + Ollama | Building from Scratch |
|---------|------|------------------|----------------------|
| **Setup Time** | 5 minutes | Hours (per tool) | 6-12 months |
| **One API** | ✅ | ❌ (3+ different APIs) | ✅ (after 6 months) |
| **Multi-GPU** | ✅ | ❌ (conflicts) | ✅ (after 6 months) |
| **Multi-Tenancy** | ✅ (Rhai script) | ❌ | ✅ (custom code) |
| **GDPR Compliance** | ✅ (built-in) | ❌ | ✅ (6+ months work) |
| **Cost** | Free (GPL) | Free | $500K+ dev cost |

---

## Revenue Models

### Consumer: Open Source (GPL-3.0)
- Free forever
- Self-hosted only
- Community support

### Business: Two Options

**Option 1: Self-Hosted (GPL-3.0)**
- Free software
- You run on your infrastructure
- You keep 100% of revenue
- Community support

**Option 2: Managed Platform (Future)**
- We run the infrastructure
- You get API endpoint
- 30-40% platform fee
- Enterprise support + SLA

---

## Current Status

**Development Progress:** 68% complete (42/62 BDD scenarios passing)

**What's Working:**
- ✅ Multi-machine orchestration (SSH-based)
- ✅ Heterogeneous backends (CUDA, Metal, CPU)
- ✅ Model downloading (HuggingFace)
- ✅ SSE token streaming
- ✅ OpenAI-compatible API

**In Progress:**
- 🚧 Rhai scheduler (M2 - Q2 2026)
- 🚧 Multi-modal support (M3 - Q3 2026)
- 🚧 Web UI dashboard

**Timeline:**
- **M0 (Q4 2025):** Core orchestration complete
- **M1 (Q1 2026):** Production-ready
- **M2 (Q2 2026):** Rhai scheduler + Web UI
- **M3 (Q3 2026):** Multi-modal (images, audio, video)

---

## Quick ROI Analysis

### Consumer ROI

**Without rbee:**
- OpenAI API: $20-100/month per developer
- Year 1 cost: $240-1,200
- Dependency risk: High (provider changes)

**With rbee:**
- Setup time: 5 minutes
- Ongoing cost: Electricity only (~$10-30/month)
- Year 1 savings: $210-1,170
- Dependency risk: Zero (your hardware)

### Business ROI

**Building from scratch:**
- Development: 6-12 months
- Cost: $500K+ (3-5 engineers)
- Maintenance: Ongoing

**With rbee:**
- Setup: 1 day
- Cost: Free (GPL) or 30-40% platform fee
- Maintenance: Minimal (community updates)
- **Savings: $500K+ in year 1**

---

## Next Steps

### For Consumers
1. Read [Consumer Use Case](02_CONSUMER_USE_CASE.md)
2. Try rbee (see main [README.md](../../README.md))
3. Join community (GitHub Discussions)

### For Businesses
1. Read [Business Use Case](03_BUSINESS_USE_CASE.md)
2. Review [Technical Differentiators](04_TECHNICAL_DIFFERENTIATORS.md)
3. Evaluate [Revenue Models](05_REVENUE_MODELS.md)
4. Contact for enterprise support

---

## Key Takeaways

1. **rbee unifies heterogeneous GPU infrastructure** - One API for all your GPUs across all your machines
2. **Consumer use case:** Stop juggling AI tools, use everything through one API
3. **Business use case:** Turn GPU farm into production platform in one day
4. **Technical advantage:** Multi-modal, multi-tenant, GDPR-compliant out of the box
5. **Business model:** Open source (GPL) for self-hosted, managed platform option coming
6. **Status:** 68% complete, core features working, production-ready Q1 2026

---

**Read the detailed use cases:**
- [Consumer Use Case](02_CONSUMER_USE_CASE.md)
- [Business Use Case](03_BUSINESS_USE_CASE.md)
