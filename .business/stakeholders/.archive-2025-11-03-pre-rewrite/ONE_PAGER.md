# rbee: One-Page Overview

**Version:** 2.0 | **Date:** November 2, 2025

---

## What is rbee?

**rbee turns heterogeneous GPU infrastructure into a unified AI platform with one simple API.**

---

## The Problem

### Consumers
You have multiple computers with different GPUs (Mac M2, RTX 4090, old server). You're juggling ComfyUI, Ollama, and Whisper. They fight over GPU memory. Different APIs. Manual switching. Chaos.

### Businesses
You have a GPU farm (20x A100, 10x H100). You want to offer AI services. But text + images + audio = 3 different platforms. Different APIs. No multi-tenancy. No GDPR compliance. 6-12 months to build from scratch.

---

## The Solution

### One API for Everything

```bash
# Consumer: 5-minute setup
rbee hive install gaming-pc    # RTX 4090
rbee hive install mac-studio   # M2 Ultra

# Now use everything through one API
curl http://localhost:7833/v1/chat/completions -d '...'  # LLM
curl http://localhost:7833/v1/images/generations -d '...' # Images
# Both run AT THE SAME TIME, no conflicts
```

### Business: One Platform, All Modalities

```bash
# Setup your GPU farm (one time)
rbee hive install gpu-node-01  # 4x A100
rbee hive install gpu-node-02  # 2x H100

# Configure multi-tenancy (Rhai script)
fn route_task(task, workers) {
    if task.customer_tier == "enterprise" {
        return workers.filter(|w| w.gpu_type == "H100").least_loaded();
    }
    return workers.filter(|w| w.gpu_type == "A100").least_loaded();
}

# Your customers get ONE API endpoint
# https://api.yourcompany.com/v1/
```

---

## Value Propositions

### For Consumers
**"Stop juggling AI tools. One API for everything."**

- ✅ Use ALL your GPUs (Mac, Windows, Linux)
- ✅ Mix CUDA, Metal, CPU freely
- ✅ Pin models to specific GPUs (GUI or script)
- ✅ OpenAI-compatible (works with Zed, Cursor)
- ✅ Free and open source (GPL-3.0)

### For Businesses
**"Turn your GPU farm into a product in one day."**

- ✅ One API for text, images, audio, video
- ✅ Multi-tenancy out of the box (Rhai scheduler)
- ✅ GDPR compliance built-in
- ✅ Save $500K-1M vs building from scratch
- ✅ Keep 100% of revenue (self-hosted)

---

## Unique Advantages

1. **Heterogeneous Hardware** - ONLY solution supporting CUDA + Metal + CPU in one cluster
2. **Rhai Programmable Scheduler** - ONLY solution with scriptable routing (no recompilation)
3. **GDPR Compliance** - ONLY open-source solution with built-in audit logs and EU enforcement
4. **SSH-Based Deployment** - ONLY solution that works without Kubernetes (homelab-friendly)

---

## Current Status

**Development:** 68% complete (42/62 BDD scenarios passing)

**What's Working:**
- ✅ Multi-machine orchestration (SSH-based)
- ✅ Heterogeneous backends (CUDA, Metal, CPU)
- ✅ OpenAI-compatible API
- ✅ SSE token streaming

**Timeline:**
- **M0 (Q4 2025):** Core orchestration complete
- **M1 (Q1 2026):** Production-ready
- **M2 (Q2 2026):** Rhai scheduler + Web UI
- **M3 (Q3 2026):** Multi-modal (images, audio, video)

---

## Revenue Models

| Model | Target | Revenue |
|-------|--------|---------|
| **Open Source** | Consumers | $0 (free forever) |
| **Self-Hosted** | Businesses | Keep 100% of revenue |
| **Managed Platform** | Businesses | 30-40% platform fee |
| **GPU Marketplace** | Providers + Customers | 30-40% platform fee |

---

## ROI

### Consumer
- **Without rbee:** $240-1,200/year (OpenAI API)
- **With rbee:** $120-360/year (electricity)
- **Savings:** $120-840/year

### Business
- **Build from scratch:** $750K-1.4M (Year 1)
- **rbee (self-hosted):** $500 (setup)
- **Savings:** $750K-1.4M (Year 1)

---

## Financial Projections

**Year 1 (Self-Hosted):**
- Customers: 100 Pro + 10 Enterprise
- Revenue: $238,680/year
- Profit: $212,280/year

**Year 3 (Self-Hosted):**
- Customers: 500 Pro + 50 Enterprise
- Revenue: $1,193,400/year
- Profit: $1,061,400/year

**Investment:** $220K-340K (M0-M4 over 18 months)  
**Expected Revenue (Year 3):** $2.4M+  
**ROI:** Positive by Year 2

---

## Comparisons

| Feature | rbee | Ollama | Build from Scratch | Together.ai |
|---------|------|--------|-------------------|-------------|
| **Multi-Machine** | ✅ | ❌ | ✅ | ✅ |
| **Multi-Modal** | ✅ (M3) | ❌ | ✅ | ⚠️ |
| **Multi-Tenancy** | ✅ Rhai | ❌ | ✅ Custom | ✅ |
| **GDPR** | ✅ Built-in | ❌ | ✅ Custom | ⚠️ |
| **Setup Time** | 5 min | 5 min | 6-12 months | 1 hour |
| **Cost** | Free | Free | $500K+ | Per-token |
| **Control** | ✅ Full | ✅ Full | ✅ Full | ❌ Limited |

---

## Next Steps

### For Consumers
1. Read [Consumer Use Case](02_CONSUMER_USE_CASE.md)
2. Try rbee (see main [README.md](../../README.md))
3. Join community (GitHub Discussions)

### For Businesses
1. Read [Business Use Case](03_BUSINESS_USE_CASE.md)
2. Review [Revenue Models](05_REVENUE_MODELS.md)
3. Evaluate [Implementation Roadmap](06_IMPLEMENTATION_ROADMAP.md)

### For Investors
1. Read [Executive Summary](01_EXECUTIVE_SUMMARY.md)
2. Review [Revenue Models](05_REVENUE_MODELS.md)
3. Review [Implementation Roadmap](06_IMPLEMENTATION_ROADMAP.md)

---

## Contact

- **GitHub:** https://github.com/veighnsche/llama-orch
- **Discussions:** GitHub Discussions
- **License:** GPL-3.0-or-later

---

**rbee: Turn your GPU farm into one unified API.** 🐝
