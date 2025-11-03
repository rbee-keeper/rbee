# rbee: Revenue Models

**Audience:** Business stakeholders, investors, founders  
**Date:** November 2, 2025

---

## Overview

rbee supports multiple revenue models depending on your use case and business goals.

---

## Model 1: Open Source (Consumer)

### Target Audience
- Homelab users
- AI enthusiasts
- Power users
- Developers

### License
**GPL-3.0-or-later** (Free and open source, copyleft)

### Revenue
**$0** - Completely free

### Value Proposition
- ✅ Use ALL your GPUs across ALL your computers
- ✅ No cloud costs (electricity only)
- ✅ Complete privacy (data never leaves your network)
- ✅ No dependency on external providers
- ✅ Free forever

### Cost Structure
**One-time:**
- Setup time: 5 minutes
- Learning curve: 1 hour

**Ongoing:**
- Electricity: ~$10-30/month (GPU power)
- Internet: $0 (local network)
- **Total: ~$120-360/year**

### Comparison to Cloud APIs
**Without rbee (OpenAI API):**
- $20-100/month per developer
- Year 1 cost: $240-1,200

**With rbee:**
- Year 1 cost: $120-360 (electricity)
- **Savings: $120-840/year**

---

## Model 2: Self-Hosted (Business)

### Target Audience
- GPU infrastructure operators
- AI service providers
- Startups with GPU farms
- Enterprises

### License
**GPL-3.0-or-later** (Free software, copyleft)

### Revenue
**You keep 100% of revenue from your customers**

### Value Proposition
- ✅ Free software (GPL-3.0)
- ✅ Run on your infrastructure
- ✅ Keep 100% of revenue
- ✅ Full control over data and models
- ✅ GDPR compliance built-in
- ✅ Multi-tenancy out of the box

### Cost Structure

**One-time:**
- Setup: 1 day (~$500 in labor)
- Learning curve: 1 week (Rhai scripting)

**Ongoing:**
- GPU infrastructure: Already owned
- Electricity: ~$0.50-1.00/GPU/hour
- Bandwidth: Minimal (API responses)
- Maintenance: Community updates (free)
- **Total: GPU electricity only**

### Example Business Model

**Your pricing tiers:**
| Tier | Monthly Price | Features |
|------|---------------|----------|
| **Free** | $0 | 10K tokens/day, llama-3-8b, 1 concurrent |
| **Pro** | $99 | 1M tokens/day, llama-3-70b, 10 concurrent |
| **Enterprise** | $999 | Unlimited, llama-3-405b, 100 concurrent, SLA |

**Your costs (10x A100 GPUs):**
- Electricity: ~$2,000/month
- Bandwidth: ~$200/month
- **Total: ~$2,200/month**

**Your revenue (100 Pro + 10 Enterprise customers):**
- Pro: $99 × 100 = $9,900/month
- Enterprise: $999 × 10 = $9,990/month
- **Total: $19,890/month**

**Your profit:**
- Revenue: $19,890/month
- Costs: $2,200/month
- **Profit: $17,690/month = $212,280/year**

---

## Model 3: Managed Platform (Future)

### Target Audience
- Businesses without GPU infrastructure
- Businesses wanting zero infrastructure management
- Businesses needing enterprise support + SLA

### License
**Commercial** (Managed service)

### Revenue
**Platform fee: 30-40% of customer revenue**

### Value Proposition
- ✅ Zero infrastructure management
- ✅ Enterprise support included
- ✅ SLA guarantees
- ✅ Automatic scaling
- ✅ GDPR compliance built-in
- ✅ Multi-tenancy out of the box

### Cost Structure

**One-time:**
- Setup: 1 hour (configuration)
- **Total: ~$100**

**Ongoing:**
- Platform fee: 30-40% of revenue
- No infrastructure costs
- No maintenance costs
- **Total: Platform fee only**

### Example Business Model

**Your pricing tiers:**
| Tier | Monthly Price | Features |
|------|---------------|----------|
| **Pro** | $99 | 1M tokens/day, llama-3-70b, 10 concurrent |
| **Enterprise** | $999 | Unlimited, llama-3-405b, 100 concurrent, SLA |

**Your revenue (100 Pro + 10 Enterprise customers):**
- Pro: $99 × 100 = $9,900/month
- Enterprise: $999 × 10 = $9,990/month
- **Total: $19,890/month**

**Platform fee (35%):**
- Fee: $19,890 × 0.35 = $6,961.50/month
- **Your net revenue: $12,928.50/month = $155,142/year**

**Trade-off:**
- ❌ Pay platform fee (35%)
- ✅ Zero infrastructure costs
- ✅ Zero maintenance
- ✅ Enterprise support + SLA

---

## Model 4: GPU Marketplace (Future)

### Concept
**Airbnb for GPUs** - Connect GPU providers with customers

### How It Works

**For GPU Providers:**
1. Run rbee on your GPU infrastructure
2. Register with marketplace platform
3. Set your pricing (per token or per hour)
4. Earn money when customers use your GPUs

**For Customers:**
1. Sign up for marketplace
2. Get API endpoint
3. Use OpenAI-compatible API
4. Pay per task (not per hour)

**For Platform (You):**
1. Match supply with demand
2. Handle billing and payments
3. Enforce SLAs and quality
4. Take 30-40% platform fee

### Revenue Model

**Provider pricing:**
- Provider sets: $0.50/1M tokens
- Platform fee: 35%
- Provider earns: $0.325/1M tokens
- Customer pays: $0.50/1M tokens

**Platform revenue:**
- Platform fee: $0.175/1M tokens (35%)
- Volume: 100B tokens/month
- **Platform revenue: $17,500/month = $210,000/year**

### Advantages

**vs RunPod/Vast.ai:**
- ✅ Task-based pricing (not hourly)
- ✅ No idle costs for customers
- ✅ Automatic orchestration
- ✅ Multi-modal support

**vs Together.ai/Replicate:**
- ✅ Your own provider network
- ✅ Control margins (30-40%)
- ✅ EU-only enforcement (GDPR)
- ✅ Custom models supported

---

## Financial Projections

### Conservative Scenario (Self-Hosted)

**Year 1:**
- Customers: 100 Pro + 10 Enterprise
- Revenue: $19,890/month = $238,680/year
- Costs: $2,200/month = $26,400/year
- **Profit: $212,280/year**

**Year 2:**
- Customers: 300 Pro + 30 Enterprise
- Revenue: $59,670/month = $716,040/year
- Costs: $6,600/month = $79,200/year
- **Profit: $636,840/year**

**Year 3:**
- Customers: 500 Pro + 50 Enterprise
- Revenue: $99,450/month = $1,193,400/year
- Costs: $11,000/month = $132,000/year
- **Profit: $1,061,400/year**

---

### Aggressive Scenario (Managed Platform)

**Year 1:**
- Customers: 500 Pro + 50 Enterprise
- Revenue: $99,450/month = $1,193,400/year
- Platform fee (35%): $417,690/year
- **Your net: $775,710/year**

**Year 2:**
- Customers: 1,500 Pro + 150 Enterprise
- Revenue: $298,350/month = $3,580,200/year
- Platform fee (35%): $1,253,070/year
- **Your net: $2,327,130/year**

**Year 3:**
- Customers: 3,000 Pro + 300 Enterprise
- Revenue: $596,700/month = $7,160,400/year
- Platform fee (35%): $2,506,140/year
- **Your net: $4,654,260/year**

---

### GPU Marketplace Scenario

**Year 1:**
- Providers: 50 (average 10 GPUs each = 500 GPUs)
- Customers: 1,000
- Volume: 10B tokens/month
- Platform fee (35%): $0.175/1M tokens
- **Platform revenue: $1,750/month = $21,000/year**

**Year 2:**
- Providers: 200 (2,000 GPUs)
- Customers: 5,000
- Volume: 100B tokens/month
- **Platform revenue: $17,500/month = $210,000/year**

**Year 3:**
- Providers: 500 (5,000 GPUs)
- Customers: 20,000
- Volume: 500B tokens/month
- **Platform revenue: $87,500/month = $1,050,000/year**

---

## ROI Comparison

### Building from Scratch vs rbee (Self-Hosted)

**Build from Scratch:**
- Development: 6-12 months
- Cost: $450K-1M (3-5 engineers)
- Ongoing: $300K-400K/year (2 engineers)
- **Total Year 1: $750K-1.4M**

**rbee (Self-Hosted):**
- Setup: 1 day
- Cost: $500 (1 engineer)
- Ongoing: $0/year (community updates)
- **Total Year 1: $500**

**Savings: $750K-1.4M in Year 1**

---

### Cloud APIs vs rbee (Self-Hosted)

**Cloud APIs (Together.ai):**
- Cost: $0.20/1M tokens
- Volume: 30M tokens/month
- **Cost: $6,000/month = $72,000/year**

**rbee (Self-Hosted):**
- GPU electricity: ~$2,000/month
- **Cost: $24,000/year**

**Savings: $48,000/year**

**Plus:**
- ✅ Keep 100% of revenue (no platform fee)
- ✅ Full control over data and models
- ✅ GDPR compliance (your infrastructure)

---

## Pricing Strategy Recommendations

### Consumer (Open Source)
**Price: $0 (Free)**

**Why:**
- Build community
- Prove product value
- Get feedback and contributions
- Establish market presence

---

### Business (Self-Hosted)

**Recommended tiers:**

| Tier | Monthly Price | Tokens/Day | Models | Concurrent | SLA |
|------|---------------|------------|--------|------------|-----|
| **Free** | $0 | 10K | llama-3-8b | 1 | None |
| **Starter** | $49 | 100K | llama-3-8b, 13b | 5 | 99% |
| **Pro** | $99 | 1M | llama-3-70b | 10 | 99.5% |
| **Business** | $499 | 10M | llama-3-70b, 405b | 50 | 99.9% |
| **Enterprise** | $999+ | Unlimited | All models | 100+ | 99.99% |

**Why this structure:**
- Free tier: Acquisition and testing
- Starter: Small businesses and individuals
- Pro: Growing businesses
- Business: Mid-market companies
- Enterprise: Large companies with custom needs

---

### Business (Managed Platform)

**Recommended tiers:**

| Tier | Monthly Price | Platform Fee | Your Net |
|------|---------------|--------------|----------|
| **Pro** | $99 | 35% ($34.65) | $64.35 |
| **Business** | $499 | 35% ($174.65) | $324.35 |
| **Enterprise** | $999+ | 30% ($299.70+) | $699.30+ |

**Why lower fee for Enterprise:**
- Incentivize large customers
- Competitive with alternatives
- Higher absolute revenue

---

## Market Opportunity

### Total Addressable Market (TAM)

**Consumer (Homelab):**
- Homelab users worldwide: ~1M
- Addressable (multi-GPU): ~100K
- Conversion rate: 10%
- **Target: 10K users**

**Business (GPU Infrastructure):**
- Companies with GPU infrastructure: ~10K
- Addressable (AI services): ~2K
- Conversion rate: 5%
- **Target: 100 customers**

---

### Serviceable Addressable Market (SAM)

**Consumer:**
- Target: 10K users
- Revenue: $0 (open source)
- **Value: Community growth**

**Business (Self-Hosted):**
- Target: 100 customers
- Average revenue: $500/month
- **SAM: $50K/month = $600K/year**

**Business (Managed Platform):**
- Target: 1,000 customers
- Average revenue: $200/month (after platform fee)
- **SAM: $200K/month = $2.4M/year**

---

### Serviceable Obtainable Market (SOM)

**Year 1:**
- Consumer: 1,000 users
- Business (Self-Hosted): 10 customers
- Business (Managed): 100 customers
- **Revenue: $20K/month = $240K/year**

**Year 2:**
- Consumer: 5,000 users
- Business (Self-Hosted): 50 customers
- Business (Managed): 500 customers
- **Revenue: $100K/month = $1.2M/year**

**Year 3:**
- Consumer: 10,000 users
- Business (Self-Hosted): 100 customers
- Business (Managed): 1,000 customers
- **Revenue: $200K/month = $2.4M/year**

---

## Monetization Timeline

### Phase 1: Open Source (Now - Q4 2025)
- **Focus:** Build community, prove product value
- **Revenue:** $0
- **Goal:** 1,000 users, 10 business customers

### Phase 2: Self-Hosted Business (Q1 2026)
- **Focus:** Business customers (self-hosted)
- **Revenue:** $20K/month
- **Goal:** 100 business customers

### Phase 3: Managed Platform (Q2 2026)
- **Focus:** Managed platform for businesses
- **Revenue:** $100K/month
- **Goal:** 500 managed customers

### Phase 4: GPU Marketplace (Q3 2026)
- **Focus:** GPU marketplace (providers + customers)
- **Revenue:** $200K/month
- **Goal:** 50 providers, 1,000 customers

---

## Key Takeaways

1. **Multiple revenue models** - Open source, self-hosted, managed, marketplace
2. **Consumer = Free** - Build community and prove value
3. **Business = Self-hosted or managed** - Choose based on infrastructure
4. **GPU marketplace = Future** - Airbnb for GPUs model
5. **ROI is massive** - $750K-1.4M savings vs building from scratch
6. **Market opportunity** - $2.4M/year SOM by Year 3

---

## Next Steps

1. **Choose your model:** Consumer, business, or both
2. **Review use cases:** [Consumer](02_CONSUMER_USE_CASE.md) or [Business](03_BUSINESS_USE_CASE.md)
3. **Evaluate alternatives:** [Technical Differentiators](04_TECHNICAL_DIFFERENTIATORS.md)
4. **Plan implementation:** [Implementation Roadmap](06_IMPLEMENTATION_ROADMAP.md)
