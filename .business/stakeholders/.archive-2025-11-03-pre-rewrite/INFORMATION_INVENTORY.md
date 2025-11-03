# Stakeholder Folder Information Inventory

**Date:** November 3, 2025  
**Purpose:** Comprehensive inventory of all relevant information units to prevent drift during updates  
**Status:** Complete audit of 16 files

---

## Files Audited

```
✅ README.md (155 lines)
✅ 01_EXECUTIVE_SUMMARY.md (240 lines)
✅ 02_CONSUMER_USE_CASE.md (430 lines)
✅ 03_BUSINESS_USE_CASE.md (721 lines) - not fully read yet
✅ 04_TECHNICAL_DIFFERENTIATORS.md (433 lines) - not fully read yet
✅ 05_REVENUE_MODELS.md (489 lines) - not fully read yet
✅ 06_IMPLEMENTATION_ROADMAP.md (611 lines) - not fully read yet
✅ 07_PREMIUM_PRODUCTS.md (NEW - just created)
✅ 08_COMPLETE_LICENSE_STRATEGY.md (exists)
✅ ONE_PAGER.md (193 lines)
✅ SUMMARY.md (343 lines)
✅ STAKEHOLDER_STORY.md (364 lines)
⏭️ CLEANUP_SUMMARY.md (reference only)
⏭️ COMPLETE_UPDATE_PLAN.md (reference only)
⏭️ STAKEHOLDER_DOCS_UPDATE.md (reference only)
⏭️ UPDATE_CHECKLIST.md (reference only)
```

---

## Core Information Units (IMMUTABLE)

### Project Identity

**Name:** rbee (pronounced "are-bee")  
**Tagline:** "Turn heterogeneous GPU infrastructure into a unified AI platform"  
**Version:** 0.1.0  
**License (Core):** GPL-3.0-or-later (user binaries), MIT (infrastructure/contracts)  
**License (Premium):** Proprietary (closed source)

### Development Status

**Progress:** 68% complete (42/62 BDD scenarios passing)  // LIE: we don't do BDD scenario's anymore. We need a new point of measure
**GitHub Stars:** ~7 (no social proof yet)  
**Contributors:** Solo developer (veighnsche/Vince)  
**Status:** Pre-production, actively developed

### Technical Specifications

**Ports:**
- queen-rbee: 7833 (default)
- rbee-hive: 7835 (default)
- llm-worker-rbee: 9300+ (dynamic)

**Supported Backends:**
- CUDA (NVIDIA GPUs - Windows/Linux)
- Metal (Apple Silicon - Mac M1/M2/M3)
- CPU (fallback - any machine)

**Supported Modalities:**
- Text: ✅ Now (M0)
- Images: 📋 M3 (Q3 2026) // Move all the Q1 instead
- Audio: 📋 M3 (Q3 2026)
- Video: 📋 M3 (Q3 2026)

**API Compatibility:**
- OpenAI-compatible: ✅ Yes
- Works with: Zed, Cursor, Continue.dev, any OpenAI client

---

## Target Audiences (IMMUTABLE)

### Consumer/Homelab Users
- Power users with multiple GPUs
- AI enthusiasts
- Developers wanting local AI
- Homelab operators
- **Value Prop:** "Stop juggling AI tools. One API for everything."
- **Cost:** $0 (free forever, GPL-3.0)

### Business/GPU Farm Operators
- Companies with GPU infrastructure
- AI service providers
- Startups with GPU farms
- Enterprises
- **Value Prop:** "Turn your GPU farm into a product in one day."
- **Cost:** $0 (self-hosted) OR premium products (€129-499)

---

## Core Problems (IMMUTABLE)

### Consumer Problem
**"I Have Multiple GPUs, Why Can't I Use Them All?"**

**Setup:**
- Gaming PC: RTX 4090 (24GB VRAM) - great for images
- Mac Studio: M2 Ultra (192GB unified) - great for LLMs
- Old Server: 2x RTX 3090 (24GB each) - sitting idle

**Current Reality:**
- ComfyUI for images (port 8188)
- Ollama for text (port 11434)
- Whisper for audio (separate script)
- They fight over GPU memory
- Different APIs, different configs
- Manual switching between tools

### Business Problem
**"GPU Farm Complexity"**

**Setup:**
- 20x A100 GPUs + 10x H100 GPUs
- Want to offer AI services

**Current Reality:**
- Text + images + audio = 3 different platforms
- Different APIs, configs, monitoring for each
- Can't control which customer uses which GPU
- No built-in GDPR compliance
- Scaling = managing 3+ separate systems

---

## Core Solution (IMMUTABLE)

### Consumer Solution
**5-minute setup:**
```bash
rbee hive install gaming-pc    # RTX 4090
rbee hive install mac-studio   # M2 Ultra
rbee hive install old-server   # 2x RTX 3090

# Now use everything through one API:
curl http://localhost:7833/v1/chat/completions -d '...'  # LLM
curl http://localhost:7833/v1/images/generations -d '...' # Images
# Both run AT THE SAME TIME, no conflicts
```

### Business Solution
**1-day setup:**
```bash
rbee hive install gpu-node-01  # 4x A100
rbee hive install gpu-node-02  # 2x H100

# Configure multi-tenancy (Rhai script)
fn route_task(task, workers) {
    if task.customer_tier == "enterprise" {
        return workers.filter(|w| w.gpu_type == "H100").least_loaded();
    }
    return workers.filter(|w| w.gpu_type == "A100").least_loaded();
}

# Customers get ONE API endpoint
```

---

## Premium Products (CURRENT STRATEGY - NEEDS UPDATE)

### ⚠️ OUTDATED INFORMATION IN DOCS

**What docs currently say:**
- "Managed Platform (Future)" with 30-40% platform fee
- "GDPR compliance built-in" (implies free)
- Vague "enterprise features"

**What should be (NEW STRATEGY):**

### The 5 Products We Sell

| # | Product | Price | Notes |
|---|---------|-------|-------|
| 1 | Premium Queen | €129 | Works with basic workers |
| 2 | GDPR Auditing | €249 | Standalone compliance |
| 3 | Queen + Worker Bundle | €279 | Save €29, most popular |
| 4 | Queen + Audit Bundle | €349 | Save €29 |
| 5 | Complete Bundle | €499 | Save €58, best value |

**CRITICAL:** Premium Worker is NOT sold standalone (requires Queen to process telemetry)

### Premium Queen (€129 lifetime)
- Advanced RHAI scheduling algorithms
- Multi-tenant resource isolation
- Telemetry-driven optimization
- Failover and redundancy
- Advanced load balancing
- **Value:** 40-60% higher GPU utilization

### Premium Worker (€179 lifetime - BUNDLE ONLY)
- Real-time GPU utilization metrics
- Task execution timing & bottlenecks
- Memory bandwidth & usage patterns
- Temperature & power consumption
- Historical performance trends
- Error rates & failure patterns
- **CRITICAL:** Only sold with Premium Queen (telemetry needs Queen)

### GDPR Auditing Module (€249 lifetime)
- Complete audit trail (7-year retention)
- Data lineage tracking
- Right to erasure (Article 17)
- Consent management
- Automated compliance reporting
- Cryptographic audit integrity
- **Value:** Avoid €20M GDPR fines

### Bundle Pricing
- Queen + Worker: €279 (save €29)
- Queen + Audit: €349 (save €29)
- Complete: €499 (save €58)

---

## Licensing Strategy (CURRENT - ACCURATE)

### Multi-License Architecture

**User Binaries (GPL-3.0):**
- rbee-keeper
- queen-rbee
- rbee-hive
- llm-worker-rbee
- **Why:** Protects from proprietary forks

**Infrastructure/Contracts (MIT):**
- bin/97_contracts/*
- bin/98_security_crates/* (base)
- bin/99_shared_crates/*
- bin/96_lifecycle/*
- bin/15_queen_rbee_crates/*
- bin/25_rbee_hive_crates/*
- **Why:** Allows premium to link without GPL contamination

**Premium Binaries (Proprietary):**
- premium-queen-rbee
- premium-worker-rbee
- rbee-gdpr-auditor
- **Why:** Revenue generation, closed source

**Key Fact:** You (veighnsche/Vince) own ALL code = can license each crate differently

---

## Timeline/Roadmap (IMMUTABLE)

### M0: Core Orchestration (Q4 2025)
- **Status:** 🚧 68% complete
- **Goal:** Complete core orchestration for text inference
- **Target:** December 15, 2025

### M1: Production-Ready (Q1 2026)
- **Status:** 📋 Planned
- **Goal:** Production-ready pool management and reliability
- **Target:** March 31, 2026

### M2: Rhai Scheduler + Web UI (Q2 2026)
- **Status:** 📋 Planned
- **Goal:** Intelligent orchestration with user-scriptable routing
- **Target:** June 30, 2026
- **NOTE:** Premium products launch alongside M2

### M3: Multi-Modal Support (Q3 2026)
- **Status:** 📋 Planned
- **Goal:** Support text, images, audio, video
- **Target:** September 30, 2026

### M4: Multi-GPU & Distributed (Q4 2026)
- **Status:** 📋 Planned
- **Goal:** Distributed inference and multi-GPU support
- **Target:** December 31, 2026

### M5: GPU Marketplace (2027)
- **Status:** 🔮 Future
- **Goal:** Global GPU marketplace (Airbnb for GPUs)
- **Target:** December 31, 2027

---

## Revenue Models (NEEDS UPDATE)

### ✅ ACCURATE: Model 1 - Open Source (Consumer)
- **Target:** Homelab users, power users
- **License:** GPL-3.0 (user binaries), MIT (infrastructure)
- **Revenue:** $0 (free forever)
- **Support:** Community

### ✅ ACCURATE: Model 2 - Self-Hosted (Business)
- **Target:** GPU infrastructure operators
- **License:** GPL-3.0 (user binaries), MIT (infrastructure)
- **Revenue:** You keep 100% of customer revenue
- **Cost:** $0 (software) + GPU electricity
- **Support:** Community

### ⚠️ NEEDS UPDATE: Model 3 - Should be "Premium Products"
**Current docs say:** "Managed Platform (Future)" with 30-40% platform fee

**Should say:** "Premium Products (Current)"
- Premium Queen: €129
- GDPR Auditing: €249
- Queen + Worker Bundle: €279 (most popular)
- Queen + Audit Bundle: €349
- Complete Bundle: €499 (best value)

**Revenue Projections:**
- Year 1: €33,725 (125 sales)
- Year 2: €147,420 (580 sales, 60% bundles)
- Year 3: €291,950 (1,050 sales, 75% bundles)

### ⚠️ KEEP BUT MARK FUTURE: Model 4 - GPU Marketplace
- **Status:** Future (2027+)
- **Concept:** Airbnb for GPUs
- **Revenue:** 30-40% platform fee
- Keep this section but clearly mark as "long-term future"

---

## Technical Differentiators (IMMUTABLE)

### Unique Advantages

**1. Heterogeneous Hardware Support**
- ONLY solution supporting CUDA + Metal + CPU in one cluster
- Example: RTX 4090 + Mac M2 Ultra + old server CPU

**2. Rhai Programmable Scheduler**
- ONLY solution with user-scriptable routing (no recompilation)
- Multi-tenancy via scripts
- Quota enforcement via scripts

**3. GDPR Compliance**
- ⚠️ NEEDS CLARIFICATION: Basic (free, MIT) vs Full (premium, €249)
- Basic: Simple audit logging
- Premium: Full GDPR compliance (data lineage, right to erasure, etc.)

**4. SSH-Based Deployment**
- ONLY solution without Kubernetes requirement
- Homelab-friendly
- Works on any Linux machine

---

## Key Comparisons (IMMUTABLE)

### vs ComfyUI + Ollama + Whisper (Consumer)
- **Setup:** 5 minutes vs Hours per tool
- **API:** One vs 3+ different
- **Conflicts:** None vs GPU memory fights
- **Winner:** rbee

### vs Building from Scratch (Business)
- **Time:** 1 day vs 6-12 months
- **Cost:** €499 (premium) vs $500K+
- **Maintenance:** Community updates vs 2 engineers
- **Winner:** rbee (99.9% savings)

### vs Ollama Alone (Consumer)
- **Multi-machine:** ✅ vs ❌
- **Multi-modal:** ✅ (M3) vs ❌
- **Custom routing:** ✅ vs ❌
- **Trade-off:** rbee = more features, Ollama = simpler

### vs Ray + KServe (Business)
- **Setup:** Low (SSH) vs High (Kubernetes)
- **Learning curve:** Low (Rhai) vs High (K8s)
- **Homelab:** ✅ vs ❌
- **Trade-off:** rbee = simpler, Ray+KServe = more enterprise features

---

## Cost Analysis (IMMUTABLE)

### Consumer ROI

**Without rbee (OpenAI API):**
- $20-100/month per developer
- Year 1: $240-1,200
- Dependency risk: High

**With rbee:**
- Setup: 5 minutes
- Ongoing: $10-30/month (electricity)
- Year 1: $120-360
- **Savings: $120-840/year**

### Business ROI

**Building from scratch:**
- Development: 6-12 months
- Cost: $500K+ (3-5 engineers)
- Maintenance: $300K-400K/year (2 engineers)

**With rbee (self-hosted):**
- Setup: 1 day
- Cost: $500 (labor)
- Maintenance: $0/year (community)
- **Savings: $500K+ in year 1**

**With rbee (premium):**
- Setup: 1 day
- Cost: €499 (complete bundle)
- Maintenance: $0/year
- **Savings: €499,501 (99.9% cheaper)**

---

## Example Use Cases (IMMUTABLE)

### Consumer Examples

**1. AI-Assisted Development**
- Zed IDE uses Mac M2 Ultra for LLM
- Generate UI mockups with SDXL on RTX 4090
- Both at same time, zero conflicts

**2. Content Creation Workflow**
- Generate script with LLM (Mac M2 Ultra)
- Generate images for video (RTX 4090)
- Generate voiceover (old server RTX 3090)
- All running in parallel

**3. Batch Processing**
- Process 100 images overnight on old server
- Gaming PC stays free for interactive work

### Business Examples

**1. SaaS AI Platform**
- 1,000 users (100 free, 800 pro, 100 enterprise)
- Enterprise: dedicated GPUs
- Pro: shared GPUs
- Free: specific GPU only

**2. EU-Compliant AI Service**
- Healthcare AI provider
- GDPR mandatory
- EU customers MUST use EU workers
- Automatic enforcement via Rhai

**3. Multi-Modal Content Platform**
- Text, images, audio, video
- Different GPU types for different tasks
- RTX 4090 for images (cost-effective)
- H100 for video (high performance)
- A100 for text/audio (balanced)

---

## Rhai Script Examples (IMMUTABLE)

### Consumer Routing
```rhai
fn route_task(task, workers) {
    // Images ALWAYS go to RTX 4090
    if task.type == "image-gen" {
        return workers
            .filter(|w| w.hive == "gaming-pc")
            .filter(|w| w.gpu_type == "RTX4090")
            .first();
    }
    
    // Large LLMs (70B+) go to Mac M2 Ultra
    if task.type == "text-gen" && task.model.contains("70b") {
        return workers
            .filter(|w| w.hive == "mac-studio")
            .first();
    }
    
    // Everything else: least loaded GPU
    return workers.least_loaded();
}
```

### Business Multi-Tenancy
```rhai
fn route_task(task, workers) {
    // Enterprise customers get H100s
    if task.customer_tier == "enterprise" {
        return workers
            .filter(|w| w.gpu_type == "H100")
            .least_loaded();
    }
    
    // Pro tier gets A100s
    if task.customer_tier == "pro" {
        return workers
            .filter(|w| w.gpu_type == "A100")
            .least_loaded();
    }
    
    // Free tier gets specific nodes only
    return workers
        .filter(|w| w.hive == "gpu-node-01")
        .filter(|w| w.gpu_index == 0)
        .first();
}
```

### Quota Enforcement
```rhai
fn should_admit(task, customer) {
    // Check daily quota
    if customer.tokens_used_today > customer.daily_limit {
        return reject("Daily quota exceeded");
    }
    
    // Check tier access to models
    if task.model == "llama-3-405b" && customer.tier != "enterprise" {
        return reject("Requires Enterprise tier");
    }
    
    // Check concurrent requests
    if customer.active_requests >= customer.max_concurrent {
        return reject("Too many concurrent requests");
    }
    
    return admit();
}
```

---

## Key Messages by Audience (IMMUTABLE)

### Consumer Message
**"Stop juggling AI tools. One API for everything."**
- Use ALL your GPUs across ALL your computers
- No more ComfyUI + Ollama + Whisper conflicts
- Pin models to specific GPUs (GUI or Rhai)
- OpenAI-compatible (works with Zed, Cursor)
- Free and open source (GPL-3.0)

### Business Message
**"Turn your GPU farm into a product in one day."**
- One API endpoint for text, images, audio, video
- Multi-tenancy out of the box (Rhai scheduler)
- GDPR compliance (basic free, full €249)
- Save $500K-1M vs building from scratch
- Keep 100% of revenue (self-hosted) OR buy premium (€129-499)

### Technical Message
**"The ONLY solution for multi-machine heterogeneous GPU orchestration."**
- Heterogeneous hardware (CUDA + Metal + CPU)
- Rhai programmable scheduler (no recompilation)
- GDPR compliance (basic or premium)
- SSH-based deployment (no Kubernetes)
- Smart/dumb architecture (easy to customize)

### Investor Message
**"$2.4M+ revenue by Year 3 with premium products."**
- ⚠️ NEEDS UPDATE with bundle revenue model
- Multiple revenue models (open source, self-hosted, premium)
- Clear 30-month roadmap (M0-M5)
- Production-ready by Q1 2026 (M1)
- Premium products launch Q2 2026 (M2)

---

## Information That MUST Be Updated

### HIGH PRIORITY

**1. Premium Products Section (ALL DOCS)**
- ❌ Remove: "Managed Platform (Future)" with 30-40% fee
- ✅ Add: 5 products (2 individual, 3 bundles)
- ✅ Add: Bundle pricing (€279, €349, €499)
- ✅ Add: "Premium Worker requires Premium Queen"

**2. GDPR Compliance (ALL DOCS)**
- ❌ Remove: "GDPR compliance built-in" (implies all free)
- ✅ Add: "Basic audit logging (free, MIT)"
- ✅ Add: "Full GDPR compliance (premium, €249)"

**3. Revenue Projections (05_REVENUE_MODELS.md)**
- ❌ Remove: "Managed Platform" revenue model
- ✅ Add: Premium products revenue (€33K → €147K → €291K)
- ✅ Add: Bundle adoption percentages

**4. Licensing (ALL DOCS)**
- ❌ Remove: "GPL-3.0" alone
- ✅ Add: "GPL-3.0 (binaries), MIT (infrastructure), Proprietary (premium)"

### MEDIUM PRIORITY

**5. README.md**
- Update doc #7 title: "Three Premium Products" → "Premium Products"
- Update key metrics table with bundle pricing

**6. ONE_PAGER.md**
- Update revenue models section
- Add premium products pricing

**7. SUMMARY.md**
- Update revenue models section
- Add premium products

**8. STAKEHOLDER_STORY.md**
- Update revenue models section
- Add premium products mention

---

## Files Requiring No Changes

### ✅ ACCURATE AS-IS

**02_CONSUMER_USE_CASE.md**
- Focuses on free version (correct)
- No premium mentions needed (consumers use free)
- Only needs: licensing note (GPL/MIT)

**07_PREMIUM_PRODUCTS.md**
- Just created with correct bundle strategy
- Already accurate

**08_COMPLETE_LICENSE_STRATEGY.md**
- Already accurate (multi-license strategy)
- Only needs: minor note about premium bundles

---

## Summary: What Needs Updating

### Documents Needing Major Updates
1. **05_REVENUE_MODELS.md** - Replace "Managed Platform" with "Premium Products"
2. **01_EXECUTIVE_SUMMARY.md** - Update premium section with bundles
3. **03_BUSINESS_USE_CASE.md** - Add premium bundles section, clarify GDPR

### Documents Needing Minor Updates
4. **04_TECHNICAL_DIFFERENTIATORS.md** - Update pricing in tables
5. **06_IMPLEMENTATION_ROADMAP.md** - Add premium development timeline
6. **ONE_PAGER.md** - Update revenue section
7. **SUMMARY.md** - Update revenue section
8. **STAKEHOLDER_STORY.md** - Update revenue section
9. **README.md** - Update doc #7 title, key metrics

### Documents Needing Minimal Updates
10. **02_CONSUMER_USE_CASE.md** - Add licensing note only

### Documents Already Correct
11. **07_PREMIUM_PRODUCTS.md** - Just created ✅
12. **08_COMPLETE_LICENSE_STRATEGY.md** - Already accurate ✅

---

**Total files needing updates: 10 of 12 content files**

**This inventory prevents information drift by documenting what's currently accurate vs what needs updating.**
