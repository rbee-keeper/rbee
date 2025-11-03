# 🐝 rbee: One-Page Overview

**Version:** 3.0 | **Date:** November 3, 2025

---

## What is rbee?

**rbee turns heterogeneous GPU infrastructure into a unified colony with one simple API.**

```
🐝 Queen (queen-rbee) → THE BRAIN - Makes routing decisions
🐝 Hives (rbee-hive) → HOUSING - Worker nodes across machines
🐝 Workers (llm-worker-rbee) → EXECUTORS - Do inference on GPUs
```

---

## The Problem

**Consumers:** Multiple computers with GPUs. Juggling ComfyUI + Ollama + Whisper. They fight over GPU memory. Different APIs, different ports. Chaos.

**Businesses:** GPU farm (20x A100, 10x H100). Want to offer AI services. But text + images + audio = 3 platforms. No multi-tenancy. No GDPR. 6-12 months to build from scratch.

---

## The Solution

### One Hive for Everything

```bash
# Consumer: 5-minute setup
rbee hive install gaming-pc    # RTX 4090
rbee hive install mac-studio   # M2 Ultra

# Now use everything through one API
curl http://localhost:7833/v1/chat/completions -d '...'
curl http://localhost:7833/v1/images/generations -d '...'
# Both run AT THE SAME TIME, no conflicts
```

### Business: One Platform, All Modalities

```bash
# Setup your colony (one time)
rbee hive install gpu-node-01  # 4x A100
rbee hive install gpu-node-02  # 2x H100

# Configure multi-tenancy (Rhai script)
fn route_task(task, workers) {
    if task.customer_tier == "enterprise" {
        return workers.filter(|w| w.gpu_type == "H100").least_loaded();
    }
    return workers.filter(|w| w.gpu_type == "A100").least_loaded();
}
```

---

## The 6 Unique Advantages (NO Competitor Has All)

1. ✅ **Multi-machine orchestration** - Not just multi-GPU on one node
2. ✅ **Heterogeneous hardware** - CUDA + Metal + CPU in ONE colony
3. ✅ **SSH-based deployment** - No Kubernetes required
4. ✅ **User-scriptable routing** - Rhai (no recompilation)
5. ✅ **GDPR compliance** - Basic free, full €249
6. ✅ **Lifetime pricing** - €499 vs $72K+/year

**No competitor offers all 6.**

---

## Current Status

- **M0 (Q4 2025):** Core orchestration - IN PROGRESS
- **M1 (Q1 2026):** Production-ready
- **M2 (Q2 2026):** RHAI + Web UI + Premium launch
- **M3 (Q1 2026):** Multi-modal (images, audio, video)

---

## Revenue Models

| Model | Target | Cost |
|-------|--------|------|
| **Free (Consumer)** | Homelab users | $0 (GPL + MIT) |
| **Free (Business)** | Self-hosted | $0 (GPL + MIT) |
| **Premium** | Businesses | €129-499 lifetime |

**Premium Products:**
- Premium Queen: €129 (advanced RHAI)
- GDPR Auditing: €249 (full compliance)
- Queen + Worker: €279 (most popular ⭐)
- Complete Bundle: €499 (best value ⭐⭐)

---

## ROI

**Consumer:**
- Cloud APIs: $240-1,200/year
- rbee: $120-360/year
- **Savings: $120-840/year**

**Business:**
- Build from scratch: $500K-1.4M
- rbee premium: €499 lifetime
- **Savings: 99.97% cheaper**

---

## Comparisons

| Feature | rbee | Ollama | vLLM | Together.ai |
|---------|------|--------|------|-------------|
| **Multi-Machine** | ✅ | ❌ | ⚠️ | ✅ |
| **Heterogeneous** | ✅ | ❌ | ❌ | ❌ |
| **Setup** | 5 min | 5 min | Complex | 1 hour |
| **Rhai Routing** | ✅ | ❌ | ❌ | ❌ |
| **GDPR Built-In** | ✅ | ❌ | ❌ | ⚠️ |
| **Cost (Year 1)** | €499 | FREE | FREE | $72K+ |

---

## Next Steps

**Consumers:** [Consumer Use Case](02_CONSUMER_USE_CASE.md)  
**Businesses:** [Business Use Case](03_BUSINESS_USE_CASE.md)  
**Comparing:** [Competitive Analysis](04_COMPETITIVE_ANALYSIS.md)  
**Investors:** [Executive Summary](01_EXECUTIVE_SUMMARY.md) + [Roadmap](06_IMPLEMENTATION_ROADMAP.md)

---

**🐝 Turn your GPU farm into a thriving hive!**
