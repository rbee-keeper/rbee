# rbee Stakeholder Documentation

**Version:** 2.0  
**Date:** November 2, 2025  
**Status:** Active

---

## Overview

rbee turns heterogeneous GPU infrastructure into a unified AI platform with one simple API.

**For Consumers:** Stop juggling multiple AI tools. Use ALL your GPUs across ALL your computers.

**For Businesses:** Turn your GPU farm into a production-ready AI platform with text, images, video, and audio.

---

## Documentation Structure

### 0. [One-Pager](ONE_PAGER.md)
**Quick reference** - 1-page overview for elevator pitches
- What rbee is (one sentence)
- The problem (consumer + business)
- The solution (code examples)
- Value propositions
- Unique advantages
- Current status & timeline
- ROI & financial projections
- Comparisons table
- Next steps

### 1. [Executive Summary](01_EXECUTIVE_SUMMARY.md)
**Read this first** - 2-page overview for decision makers
- What rbee is
- Core value propositions
- Consumer vs Business use cases
- Quick ROI analysis

### 2. [Consumer Use Case](02_CONSUMER_USE_CASE.md)
**For homelab users and power users**
- The multi-GPU juggling problem
- How rbee solves it
- Setup walkthrough (5 minutes)
- GUI vs Rhai scripting
- Real-world examples

### 3. [Business Use Case](03_BUSINESS_USE_CASE.md)
**For GPU infrastructure operators**
- The platform complexity problem
- One API for all modalities
- Multi-tenancy & quotas
- GDPR compliance
- Custom model catalogs

### 4. [Technical Differentiators](04_TECHNICAL_DIFFERENTIATORS.md)
**Why rbee vs alternatives**
- vs ComfyUI + Ollama + Whisper
- vs Building from scratch
- vs Cloud APIs (OpenAI, Anthropic)
- vs GPU rental platforms (RunPod, Vast.ai)

### 5. [Revenue Models](05_REVENUE_MODELS.md)
**Business models and pricing**
- **NEW DECISION:** Core = FREE FOREVER | Premium = Paid
- Consumer: Core rbee (GPL-3.0, $0 forever)
- Business: Premium products (€129-249 lifetime)
- Platform marketplace model (future)
- Financial projections

### 6. [Implementation Roadmap](06_IMPLEMENTATION_ROADMAP.md)
**What's ready now vs future**
- Current status (68% complete)
- M0: Core orchestration (Q4 2025)
- M1: Production-ready (Q1 2026)
- M2: Rhai scheduler (Q2 2026)
- M3: Multi-modal (Q3 2026)

### 7. [Three Premium Products](07_THREE_PREMIUM_PRODUCTS.md) ⭐ NEW
**THE 3 THINGS WE'RE SELLING (Critical!)**
- **MOST IMPORTANT:** Core rbee = FREE FOREVER (GPL-3.0)
- **Premium Product #1:** Premium Queen (€129 lifetime) - Advanced RHAI scheduling
- **Premium Product #2:** Premium Worker (€179 lifetime) - Deep telemetry, 40-60% better utilization
- **Premium Product #3:** GDPR Auditing Module (€249 lifetime) - Avoid €20M fines
- License architecture (Core = Open Source | Premium = Proprietary)

### 8. [Complete License Strategy](08_COMPLETE_LICENSE_STRATEGY.md) ⭐ SINGLE SOURCE OF TRUTH
**Multi-license architecture - everything you need**
- GPL for user binaries (protects from forks)
- MIT for infrastructure/contracts (prevents contamination!)
- Proprietary for premium (revenue)
- Per-crate licensing guide
- Automated migration script
- Implementation timeline (1 week)
- License compatibility chart
- **Follow this document for all licensing decisions**

---

## Quick Reference

### Key Metrics

| Metric | Value |
|--------|-------|
| **Development Progress** | 68% (42/62 BDD scenarios) |
| **Supported Backends** | CUDA, Metal, CPU |
| **Supported Modalities** | Text (now), Images/Audio/Video (M3) |
| **License (Binaries)** | GPL-3.0 (FREE FOREVER) |
| **License (Infrastructure)** | MIT (permissive) |
| **License (Premium)** | Proprietary (€129-249 lifetime) |
| **Target Markets** | Consumers (free), Businesses (premium) |

### Key Features

**Consumer:**
- ✅ Multi-machine GPU orchestration
- ✅ Heterogeneous hardware (CUDA, Metal, CPU)
- ✅ GUI for pinning models to GPUs
- ✅ Rhai scripting for custom routing
- ✅ OpenAI-compatible API

**Business:**
- ✅ Multi-tenancy out of the box
- ✅ GDPR compliance built-in
- ✅ Custom model catalogs
- ✅ Quota enforcement
- ✅ Cost control via Rhai scheduler

---

## For Different Audiences

### If you're a **Consumer/Homelab User**
→ Read: [Executive Summary](01_EXECUTIVE_SUMMARY.md) → [Consumer Use Case](02_CONSUMER_USE_CASE.md)

### If you're a **Business/GPU Farm Operator**
→ Read: [Executive Summary](01_EXECUTIVE_SUMMARY.md) → [Business Use Case](03_BUSINESS_USE_CASE.md)

### If you're **Evaluating Alternatives**
→ Read: [Technical Differentiators](04_TECHNICAL_DIFFERENTIATORS.md)

### If you're **Planning Investment/Budget**
→ Read: [Revenue Models](05_REVENUE_MODELS.md) → [Implementation Roadmap](06_IMPLEMENTATION_ROADMAP.md)

---

## Archive

Previous stakeholder documents are in [`.archive/`](.archive/) for historical reference.

---

**Questions?** See the main [README.md](../../README.md) or [Architecture Documentation](../../.arch/README.md)
