# rbee: Executive Summary

**Version:** 3.0  
**Date:** November 3, 2025  
**Audience:** Decision makers, investors, technical leaders

---

## What is rbee?

**rbee turns heterogeneous GPU infrastructure into a unified colony with one simple API.**

```
🐝 The rbee Colony Architecture

Beekeeper (rbee-keeper) ← Manages the colony
    ↓
Queen (queen-rbee) ← THE BRAIN - Makes ALL routing decisions
    ↓
Hives (rbee-hive) ← HOUSING - Worker nodes across machines  
    ↓
Workers (llm-worker-rbee) ← EXECUTORS - Do inference on GPUs
```

**For Consumers:** Stop juggling multiple AI tools (ComfyUI, Ollama, Whisper). Use ALL your GPUs across ALL your computers with one API.

**For Businesses:** Turn your GPU farm into a production-ready AI platform offering text, images, video, and audio from one endpoint.

---

## The Core Problem

### Consumer Problem: "I Have Multiple GPUs, Why Can't I Use Them All?"

You have:
- Gaming PC with RTX 4090 (24GB VRAM) - great for images
- Mac Studio with M2 Ultra (192GB unified) - great for LLMs
- Old server with 2x RTX 3090 (24GB each) - sitting idle

**Current reality:**
- ComfyUI for images (port 8188)
- Ollama for text (port 11434)
- Whisper for audio (separate script)
- They **fight over GPU memory**
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

### One Hive for Everything 🐝

```bash
# Consumer: 5-minute setup
rbee hive install gaming-pc    # Hive 1: RTX 4090
rbee hive install mac-studio   # Hive 2: M2 Ultra
rbee hive install old-server   # Hive 3: 2x RTX 3090

# Now the queen orchestrates ALL worker bees across ALL hives
curl http://localhost:7833/v1/chat/completions \
  -d '{"model": "llama-3-70b", "messages": [...]}'
# ↑ Queen routes to Mac M2 Ultra

curl http://localhost:7833/v1/images/generations \
  -d '{"model": "sdxl", "prompt": "a cat"}'
# ↑ Queen routes to RTX 4090

# Both run AT THE SAME TIME - No conflicts!
```

### Business: Turn Your Farm Into a Thriving Hive 🐝

```bash
# Setup your colony (one time)
rbee hive install gpu-node-01  # Hive with 4x A100
rbee hive install gpu-node-02  # Hive with 2x H100

# Configure the queen's routing (Rhai script)
fn route_task(task, workers) {
    // Enterprise customers get H100 workers
    if task.customer_tier == "enterprise" {
        return workers.filter(|w| w.gpu_type == "H100").least_loaded();
    }
    // Free tier gets A100 workers
    return workers.filter(|w| w.gpu_type == "A100").least_loaded();
}

# Your customers get ONE API endpoint:
# https://api.yourcompany.com/v1/
```

---

## The 6 Unique Advantages (NO Competitor Has All)

**rbee is the ONLY solution with ALL 6:**

### 1. ✅ Multi-Machine Orchestration
Not just multi-GPU on one node. Multiple machines with SSH-based deployment.

**Example:** Use your gaming PC + Mac + server together as ONE colony.

**Competitors:** Ollama (single machine), vLLM (single model), LocalAI (single machine)

---

### 2. ✅ Heterogeneous Hardware
CUDA + Metal + CPU in ONE colony.

**Example:** Mix NVIDIA GPUs + Apple Silicon + CPU workers in the same colony.

**Competitors:** vLLM (CUDA only), Ray+KServe (CUDA only), Together.ai (CUDA only)

---

### 3. ✅ SSH-Based Deployment
No Kubernetes required. Works on any Linux machine.

**Example:** `rbee hive install hostname` - Done. No K8s, no Docker complexity.

**Competitors:** Ray+KServe (requires K8s), vLLM (requires containers for multi-node)

---

### 4. ✅ User-Scriptable Routing
Rhai scripts (no recompilation, no restart).

**Example:** Change routing logic in seconds. Business users can modify, not just engineers.

**Competitors:** vLLM (write Python, redeploy), Ray+KServe (write K8s YAML, redeploy)

---

### 5. ✅ GDPR Compliance
Basic (free, MIT). Full (premium, €249).

**Example:** Avoid €20M GDPR fines. EU-ready Day 1.

**Competitors:** vLLM (build it yourself), Ray+KServe (build it yourself), Together.ai (trust them)

---

### 6. ✅ Lifetime Pricing
Pay once, own forever.

**Example:** €499 (complete bundle) vs $72K+/year (Together.ai)

**Competitors:** Together.ai ($0.20-3.50/M tokens), RunPod ($0.19-3.19/hour)

---

## Revenue Models

### Consumer: Free Forever 🐝

**License:** GPL-3.0 (binaries) + MIT (infrastructure)

**Cost:** $0 - Free forever

**What you get:**
- ✅ Multi-machine orchestration
- ✅ Heterogeneous hardware support
- ✅ Basic routing (Rhai scripts)
- ✅ OpenAI-compatible API
- ✅ Community support

**Perfect for:** Homelab users, power users, AI enthusiasts

---

### Business: Two Options

#### Option 1: Self-Hosted (Free) 🐝

**License:** GPL-3.0 (binaries) + MIT (infrastructure)

**Cost:** $0 (software) + GPU electricity

**What you get:**
- ✅ Multi-tenancy (Rhai scripts)
- ✅ Basic audit logging (MIT)
- ✅ Quota enforcement
- ✅ Custom routing
- ✅ Keep 100% of revenue

**What you don't get:**
- ❌ Advanced RHAI scheduler
- ❌ Deep telemetry
- ❌ Full GDPR compliance

**Perfect for:** Startups, small businesses, self-hosting enthusiasts

---

#### Option 2: Premium Products (€129-499 lifetime) 🐝

**The 5 products we sell:**

| Product | Price | What You Get |
|---------|-------|--------------|
| **Premium Queen** | €129 | Advanced RHAI scheduling (works with basic workers) |
| **GDPR Auditing** | €249 | Full compliance (standalone) |
| **Queen + Worker** | **€279** | Full smart scheduling ⭐ MOST POPULAR |
| **Queen + Audit** | €349 | Scheduling + compliance |
| **Complete Bundle** | **€499** | Everything ⭐⭐ BEST VALUE |

**Why bundles?** Premium Worker collects telemetry that Premium Queen uses for intelligent routing. Worker without Queen = useless data collection.

**Perfect for:** Businesses with 10+ GPUs, enterprises, EU businesses needing GDPR

[Full details → Premium Products](05_PREMIUM_PRODUCTS.md)

---

## Technical Differentiators

### vs Ollama (Consumer)
| Feature | rbee | Ollama |
|---------|------|--------|
| Multi-machine | ✅ | ❌ (single machine) |
| Heterogeneous | ✅ CUDA+Metal+CPU | ❌ CUDA or Metal |
| Custom routing | ✅ Rhai | ❌ |
| Multi-modal | ✅ (Q1 2026) | ❌ (text only) |

**Verdict:** Ollama for ONE machine. rbee for MULTIPLE machines.

---

### vs vLLM (Business)
| Feature | rbee | vLLM |
|---------|------|------|
| Multi-machine orchestration | ✅ | ❌ (library for one model) |
| Custom routing | ✅ Rhai (no redeploy) | ❌ Python code (redeploy) |
| Multi-tenancy | ✅ Built-in | ❌ Build yourself |
| GDPR | ✅ Built-in | ❌ Build yourself |

**Verdict:** vLLM for ONE model. rbee for MANY models across MANY machines.

---

### vs Ray + KServe (Enterprise)
| Feature | rbee | Ray + KServe |
|---------|------|--------------|
| Setup time | 1 day | 1 week (K8s) |
| Requires K8s | ❌ | ✅ |
| Learning curve | Low (Rhai) | High (K8s) |
| Homelab-friendly | ✅ | ❌ |

**Verdict:** Ray+KServe for K8s experts. rbee for everyone else.

---

### vs Together.ai (Cloud)
| Feature | rbee | Together.ai |
|---------|------|-------------|
| Year 1 cost | €499 | $72K+ |
| Privacy | ✅ Your network | ❌ Their servers |
| Control | ✅ Full | ❌ Limited |
| GDPR | ✅ Your infrastructure | ⚠️ Trust them |

**Verdict:** Together.ai for low volume (<100M tokens/month). rbee for high volume (300M+ tokens/month) or privacy needs.

---

## Current Status

**Progress:** M0 in progress (text inference working)

**What's Working:**
- ✅ Multi-machine orchestration (SSH-based)
- ✅ Heterogeneous backends (CUDA, Metal, CPU)
- ✅ Model downloading (HuggingFace)
- ✅ SSE token streaming
- ✅ OpenAI-compatible API
- ✅ Worker spawning and lifecycle

**Timeline:**
- **M0 (Q4 2025):** Core orchestration complete (text)
- **M1 (Q1 2026):** Production-ready (reliability + monitoring)
- **M2 (Q2 2026):** RHAI scheduler + Web UI + **Premium Launch** 🐝
- **M3 (Q1 2026):** Multi-modal (images, audio, video)
- **M4 (Q4 2026):** Multi-GPU & distributed
- **M5 (2027):** GPU marketplace

[Full roadmap → Implementation Roadmap](06_IMPLEMENTATION_ROADMAP.md)

---

## ROI Analysis

### Consumer ROI

**Without rbee (OpenAI API):**
- $20-100/month per developer
- Year 1: $240-1,200
- Dependency risk: High

**With rbee:**
- Setup: 5 minutes
- Ongoing: $10-30/month (electricity only)
- Year 1: $120-360
- **Savings: $120-840/year**

**Plus:**
- ✅ Complete privacy (data never leaves your network)
- ✅ No rate limits
- ✅ Use hardware you already own

---

### Business ROI

#### vs Building from Scratch

**Build from Scratch:**
- Development: 6-12 months
- Cost: $500K-1.4M (3-5 engineers × $150K-200K/year)
- Maintenance: $300K-400K/year (2 engineers ongoing)
- **Year 1 total: $800K-1.4M**

**rbee (self-hosted):**
- Setup: 1 day
- Cost: $500 (labor)
- Maintenance: $0/year (community updates)
- **Year 1 total: $500**

**rbee (premium):**
- Setup: 1 day
- Cost: €499 (one-time, lifetime)
- Maintenance: $0/year
- **Year 1 total: €499**

**Savings: $799,501-1.4M (99.97% cheaper)**

---

#### vs Together.ai (Cloud)

**Together.ai (300M tokens/month):**
- Cost: $0.88/1M tokens (Llama 3.1 70B)
- Monthly: $264
- **Year 1: $3,168**

**rbee (self-hosted):**
- GPU electricity: ~$2,000/month
- **Year 1: $24,000**

**At low volume (<100M tokens/month):** Together.ai cheaper

**At high volume (300M+ tokens/month):** rbee cheaper

**BUT - Privacy & Control:**
- ✅ rbee: Data NEVER leaves your network
- ✅ rbee: Use ANY model (your fine-tuned models)
- ✅ rbee: Full GDPR control
- ✅ rbee: No vendor lock-in

---

## Revenue Projections (Premium Products)

### Year 1 (Conservative)
- 30 Queen solo × €129 = €3,870
- 20 Audit solo × €249 = €4,980
- 40 Queen + Worker × €279 = €11,160 (most popular)
- 25 Queen + Audit × €349 = €8,725
- 10 Complete Bundle × €499 = €4,990
- **Total: €33,725**

### Year 2 (Growth - 60% bundle adoption)
- 200 Queen solo × €129 = €25,800
- 80 Audit solo × €249 = €19,920
- 150 Queen + Worker × €279 = €41,850
- 100 Queen + Audit × €349 = €34,900
- 50 Complete Bundle × €499 = €24,950
- **Total: €147,420**

### Year 3 (Established - 75% bundle adoption)
- 300 Queen solo × €129 = €38,700
- 100 Audit solo × €249 = €24,900
- 300 Queen + Worker × €279 = €83,700
- 200 Queen + Audit × €349 = €69,800
- 150 Complete Bundle × €499 = €74,850
- **Total: €291,950**

**Investment:** €33K → €147K → €291K over 3 years

[Full details → Premium Products](05_PREMIUM_PRODUCTS.md)

---

## Next Steps

### For Consumers
1. Read [Consumer Use Case](02_CONSUMER_USE_CASE.md)
2. Try rbee (see main [README.md](../../README.md))
3. Join community (GitHub Discussions)

### For Businesses
1. Read [Business Use Case](03_BUSINESS_USE_CASE.md)
2. Review [Competitive Analysis](04_COMPETITIVE_ANALYSIS.md)
3. Evaluate [Premium Products](05_PREMIUM_PRODUCTS.md)
4. Contact for enterprise support

### For Investors
1. Review [Implementation Roadmap](06_IMPLEMENTATION_ROADMAP.md)
2. Study [Premium Products](05_PREMIUM_PRODUCTS.md)
3. Analyze [Competitive Analysis](04_COMPETITIVE_ANALYSIS.md)

---

## Key Takeaways

1. **rbee unifies heterogeneous GPU infrastructure** - One colony for all your GPUs across all your machines
2. **ONLY solution with all 6 advantages** - Multi-machine + heterogeneous + SSH + Rhai + GDPR + lifetime pricing
3. **Consumer: Free forever** - GPL-3.0 + MIT (no catch, no limits)
4. **Business: Free or premium** - Self-hosted free, premium €129-499 lifetime
5. **99.97% cheaper than building** - €499 vs $500K+ from scratch
6. **Production-ready Q1 2026** - M1 milestone, premium launch Q2 2026

---

**🐝 Welcome to the rbee colony!**
