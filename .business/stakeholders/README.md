# 🐝 rbee: Stakeholder Documentation

**Version:** 3.0  
**Date:** November 3, 2025  
**Status:** Complete rewrite with bee theme

---

## What is rbee?

**rbee** (pronounced "are-bee") turns heterogeneous GPU infrastructure into a **unified colony** with one simple API.

```
🐝 The rbee Colony Architecture

Beekeeper (rbee-keeper) ← Manages everything
    ↓
Queen (queen-rbee) ← THE BRAIN - Makes all decisions
    ↓
Hives (rbee-hive) ← HOUSING - Worker nodes across machines
    ↓
Workers (llm-worker-rbee) ← EXECUTORS - Do the actual work
```

---

## Quick Start

### For Everyone
**Start here →** [Executive Summary](01_EXECUTIVE_SUMMARY.md) (2 pages)

**Quick reference →** [One-Pager](ONE_PAGER.md) (1 page)

---

## By Audience

### 🏠 Consumers & Homelab Users
**Read:** [Consumer Use Case](02_CONSUMER_USE_CASE.md)

**You'll learn:**
- How to use ALL your GPUs across ALL your computers
- 5-minute setup guide
- Stop juggling ComfyUI + Ollama + Whisper
- Rhai scripting for custom routing
- **Cost:** FREE (GPL-3.0 + MIT)

---

### 💼 Businesses & GPU Operators
**Read:** [Business Use Case](03_BUSINESS_USE_CASE.md)

**You'll learn:**
- Turn your GPU farm into a product in 1 day
- Multi-tenancy with Rhai scripts
- GDPR compliance (basic free, full €249)
- Premium products (€129-499 lifetime)
- **Cost:** FREE or Premium

---

### 🔍 Evaluating Alternatives
**Read:** [Competitive Analysis](04_COMPETITIVE_ANALYSIS.md)

**Compare rbee vs:**
- Ollama (consumer)
- vLLM (business)
- Ray + KServe (enterprise)
- Together.ai / Replicate (cloud)
- Building from scratch

**rbee's advantage:** ONLY solution with all 6 unique features

---

### 💰 Investors & Decision Makers
**Read these in order:**
1. [Executive Summary](01_EXECUTIVE_SUMMARY.md) - 2-page overview
2. [Premium Products](05_PREMIUM_PRODUCTS.md) - What we sell
3. [Implementation Roadmap](06_IMPLEMENTATION_ROADMAP.md) - Timeline & milestones

**Key numbers:**
- €33K → €147K → €291K revenue (Years 1-3)
- Production-ready: Q1 2026
- Premium launch: Q2 2026

---

## Deep Dives

### 🎯 [Premium Products](05_PREMIUM_PRODUCTS.md)
**The 5 things we sell:**
1. Premium Queen (€129) - Advanced RHAI scheduling
2. GDPR Auditing (€249) - Full compliance
3. Queen + Worker Bundle (€279) - Most popular ⭐
4. Queen + Audit Bundle (€349)
5. Complete Bundle (€499) - Best value ⭐⭐

---

### 📋 [License Strategy](07_COMPLETE_LICENSE_STRATEGY.md)
**Multi-license architecture:**
- GPL-3.0: User binaries (free forever)
- MIT: Infrastructure/contracts (prevents contamination)
- Proprietary: Premium binaries (revenue)

---

### 🗺️ [Implementation Roadmap](06_IMPLEMENTATION_ROADMAP.md)
**M0 → M5 timeline:**
- M0 (Q4 2025): Core orchestration
- M1 (Q1 2026): Production-ready
- M2 (Q2 2026): RHAI + Web UI + Premium launch
- M3 (Q1 2026): Multi-modal
- M4 (Q4 2026): Multi-GPU & distributed
- M5 (2027): GPU marketplace

---

### 🔬 [Competitive Analysis](04_COMPETITIVE_ANALYSIS.md)
**vs 8 major competitors:**
- Detailed feature comparisons
- Cost analysis
- When to choose rbee (and when not to)

---

## Key Metrics

| Metric | Value |
|--------|-------|
| **Current Status** | M0 in progress (text inference working) |
| **Supported Backends** | CUDA, Metal, CPU |
| **Supported Modalities** | Text (now), Images/Audio/Video (Q1 2026) |
| **License (Core)** | GPL-3.0 (binaries), MIT (infrastructure) |
| **License (Premium)** | Proprietary (€129-499 lifetime) |
| **Target Markets** | Consumers (free), Businesses (free or premium) |
| **Ports** | Queen: 7833, Hive: 7835, Workers: 9300+ |

---

## The 6 Unique Advantages

**rbee is the ONLY solution with ALL 6:**

1. ✅ **Multi-machine orchestration** - Not just multi-GPU on one node
2. ✅ **Heterogeneous hardware** - CUDA + Metal + CPU in ONE colony
3. ✅ **SSH-based deployment** - No Kubernetes required
4. ✅ **User-scriptable routing** - Rhai (no recompilation)
5. ✅ **GDPR compliance** - Basic free, full €249
6. ✅ **Lifetime pricing** - €499 vs $72K+/year

**No competitor offers all 6.**

---

## Key Messages

### Consumer
**"Stop juggling AI tools. One hive for everything."** 🐝
- Use ALL your GPUs across ALL your computers
- No more ComfyUI + Ollama + Whisper conflicts
- 5-minute setup, OpenAI-compatible

### Business
**"Turn your GPU farm into a thriving hive in one day."** 🐝
- Multi-tenant orchestration (Rhai scripts)
- GDPR compliance (basic free, full €249)
- Save $500K+ vs building from scratch

### Technical
**"The Queen makes all decisions. Workers just execute."** 🐝
- Queen = THE BRAIN (easy to customize)
- Workers = EXECUTORS (simple, scalable)
- Hives = HOUSING (SSH-based deployment)

---

## FAQ

**Q: What makes rbee different?**  
A: We're the ONLY multi-machine + heterogeneous solution. Plus Rhai routing, GDPR built-in, lifetime pricing.

**Q: Do I need Kubernetes?**  
A: NO! rbee uses SSH (like Ansible). Homelab-friendly.

**Q: Do I need multiple computers?**  
A: No! Works great with one computer with multiple GPUs.

**Q: Is rbee free?**  
A: YES! Core is GPL-3.0 + MIT (free forever). Premium optional (€129-499 lifetime).

**Q: When is rbee production-ready?**  
A: M1 milestone (Q1 2026)

**More questions →** [FAQ.md](FAQ.md)

---

## Archive

Previous versions archived in [`.archive-2025-11-03-pre-rewrite/`](.archive-2025-11-03-pre-rewrite/)

---

**🐝 Welcome to the rbee colony!**
