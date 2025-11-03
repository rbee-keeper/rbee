# 🐝 rbee: Business Use Case

**Audience:** GPU infrastructure operators, AI service providers, enterprises  
**Date:** November 3, 2025  
**Version:** 3.0

---

## The Problem: "Turn Your GPU Farm Into a Product"

### Your Situation

You're a business with GPU infrastructure:
- **20x NVIDIA A100 GPUs** (datacenter)
- **10x H100 GPUs** (high-performance)  
- Mix of different models and capabilities
- **Want to offer AI services to customers**

---

### Current Reality: Platform Complexity Hell

**You want to offer AI services, but:**
- ❌ Text + images + audio = 3 different platforms to manage
- ❌ Each platform has different APIs, configs, monitoring
- ❌ Customers need different API keys for each service
- ❌ You can't easily control which customer uses which GPU
- ❌ No built-in GDPR compliance (EU market blocked)
- ❌ Scaling means managing 3+ separate systems

**The cost:**
- 6-12 months development ($500K-1.4M)
- 2 engineers ongoing maintenance ($300K-400K/year)
- Complex infrastructure (K8s, monitoring, billing)
- Delayed time-to-market (competitors win)

---

## The rbee Solution: Turn Your Farm Into a Thriving Hive 🐝

### One Platform. One API. All Modalities.

```bash
# Setup (one day):
rbee hive install gpu-node-01  # Hive with 4x A100
rbee hive install gpu-node-02  # Hive with 4x A100
rbee hive install gpu-node-03  # Hive with 2x H100
# ... repeat for all nodes

# Configure your model catalog:
rbee model add llama-3-405b --hive gpu-node-03  # H100 for large models
rbee model add sdxl --hive gpu-node-01          # A100 for images
rbee model add whisper-large --hive gpu-node-02 # A100 for audio

# Set up queen's routing (Rhai script):
cat > ~/.config/rbee/scheduler.rhai << 'EOF'
fn route_task(task, workers) {
    // Enterprise customers get H100 workers
    if task.customer_tier == "enterprise" {
        return workers.filter(|w| w.gpu_type == "H100").least_loaded();
    }
    
    // Free tier gets A100 workers only
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

**Your colony is now:**
- 🐝 3 hives (3 nodes)
- 🐝 30 GPUs (30 potential worker bees)
- 🐝 1 queen (orchestrating everything)
- 🐝 1 API (one endpoint for all customers)

---

## Free Version vs Premium

### Free (Self-Hosted) 🐝

**License:** GPL-3.0 (binaries) + MIT (infrastructure)

**What you get:**
- ✅ Multi-tenant orchestration (Rhai scripts)
- ✅ Basic audit logging (MIT license)
- ✅ Quota enforcement
- ✅ Custom routing
- ✅ Keep 100% of revenue

**What you don't get:**
- ❌ Advanced RHAI scheduler
- ❌ Deep telemetry
- ❌ Full GDPR compliance

**Cost:** $0 (software) + GPU electricity

**Perfect for:** Startups, small businesses, self-hosting

---

### Premium Products (€129-499 lifetime) 🐝

**Why Premium?**
- 40-60% higher GPU utilization (Premium Queen + Worker)
- Avoid €20M GDPR fines (GDPR Auditing)
- Pay once, own forever (no recurring fees)

---

#### The 5 Products We Sell

| Product | Price | What You Get | Who It's For |
|---------|-------|--------------|--------------|
| **Premium Queen** | €129 | Advanced RHAI scheduling | Businesses with existing monitoring |
| **GDPR Auditing** | €249 | Full compliance | EU businesses, healthcare, finance |
| **Queen + Worker** | **€279** | Full smart scheduling | **MOST POPULAR** ⭐ |
| **Queen + Audit** | €349 | Scheduling + compliance | Businesses needing both |
| **Complete Bundle** | **€499** | Everything | **BEST VALUE** ⭐⭐ |

---

#### 1. Premium Queen (€129 lifetime)

**What it does:**
- Advanced RHAI scheduling algorithms
- Multi-tenant resource isolation  
- Telemetry-driven optimization
- Failover and redundancy
- Advanced load balancing

**Value:** 40-60% higher GPU utilization through intelligent task placement

**Works with:** Basic workers (no Premium Worker required)

**Who needs this:** Businesses that already have monitoring/telemetry infrastructure

---

#### 2. GDPR Auditing Module (€249 lifetime)

**What it does:**
- Complete audit trail (7-year retention)
- Data lineage tracking
- Right to erasure (Article 17)
- Consent management
- Automated compliance reporting
- Cryptographic audit integrity

**Value:** Avoid €20M GDPR fines - one fine avoided pays for this 80,000× over

**Works with:** Any rbee setup (independent of Queen/Worker)

**Who needs this:** EU businesses, healthcare providers, financial services

---

#### 3. Queen + Worker Bundle (€279 lifetime) ⭐ MOST POPULAR

**Save €29 vs buying separately (€129 + €179 = €308)**

**What's included:**
- Premium Queen (€129 value)
- Premium Worker (€179 value)

**What you get:**
- Advanced RHAI scheduling
- Deep telemetry collection
- Real-time GPU metrics
- Task execution timing
- Memory bandwidth analysis
- Temperature & power monitoring
- Historical performance trends
- Error rates & failure patterns

**Value:** 40-60% higher GPU utilization through data-driven scheduling

**Why bundle?** Premium Worker collects telemetry that Premium Queen uses for intelligent routing.

**Without Premium Queen:** Telemetry has nowhere to go = useless  
**With Premium Queen:** Telemetry enables smart routing = 40-60% better utilization

**Who needs this:** Anyone serious about GPU orchestration, businesses with 10+ GPUs

---

#### 4. Queen + Audit Bundle (€349 lifetime)

**Save €29 vs buying separately (€129 + €249 = €378)**

**What's included:**
- Premium Queen (€129 value)
- GDPR Auditing (€249 value)

**Who needs this:** EU businesses needing both scheduling and compliance

---

#### 5. Complete Bundle (€499 lifetime) ⭐⭐ BEST VALUE

**Save €58 vs buying separately (€129 + €179 + €249 = €557)**

**What's included:**
- Premium Queen (€129 value)
- Premium Worker (€179 value)
- GDPR Auditing (€249 value)

**What you get:**
Everything - full platform capabilities:
- Advanced RHAI scheduling
- Deep telemetry collection
- Complete GDPR compliance
- 40-60% higher GPU utilization
- Full EU compliance
- Data-driven decisions

**Who needs this:** Enterprise customers, EU businesses with GPU farms

**ROI:** Investment: €499 (one-time, lifetime). On €10,000 GPU hardware: €4,000-6,000 value/year. **Pays for itself in ~1 month**

---

## Business Features Out of the Box

### 1. Multi-Tenancy & Quotas (Rhai Scripts)

```rhai
// 🐝 The Queen Enforces Business Rules

fn should_admit(task, customer) {
    // Check quota
    if customer.tokens_used_today > customer.daily_limit {
        return reject("Quota exceeded");
    }
    
    // Check tier access
    if task.model == "llama-3-405b" && customer.tier \!= "enterprise" {
        return reject("Upgrade to enterprise");
    }
    
    return admit();
}

fn route_task(task, workers) {
    // Enterprise customers get H100 worker bees
    if task.customer_tier == "enterprise" {
        return workers.filter(|w| w.gpu_type == "H100").least_loaded();
    }
    
    // Free tier gets A100 worker bees
    return workers.filter(|w| w.gpu_type == "A100").least_loaded();
}
```

---

### 2. GDPR Compliance

**Basic (Free - MIT license):**
- Simple append-only logs
- Basic audit trail

**Full (Premium - €249):**
- Immutable audit logs (7-year retention)
- Data export endpoints (`/gdpr/export`)
- Data deletion endpoints (`/gdpr/delete`)
- EU-only worker filtering
- Consent tracking
- Cryptographic integrity (hash chains)

**Example:**
```bash
# Article 15: Right to access
curl https://api.yourcompany.com/gdpr/export \
  -H "Authorization: Bearer customer-key" \
  > customer-data-export.json

# Article 17: Right to erasure
curl -X DELETE https://api.yourcompany.com/gdpr/delete \
  -H "Authorization: Bearer customer-key"
```

---

### 3. Cost Control (Rhai Scripts)

```rhai
// 🐝 The Queen Optimizes Costs

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

---

### 4. Custom Model Catalog

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

## ROI Analysis

### Free Version

**Build from Scratch:**
- Development: 6-12 months
- Cost: $500K-1.4M (3-5 engineers)
- Maintenance: $300K-400K/year (2 engineers)

**rbee (self-hosted):**
- Setup: 1 day
- Cost: $500 (labor)
- Maintenance: $0/year (community)
- **Savings: $500K+ in Year 1**

---

### Premium Version

**Build from Scratch:**
- Development: 6-12 months ($500K-1.4M)
- Maintenance: $300K-400K/year
- **3-year total: $1.4M-2.2M**

**rbee (premium):**
- Setup: 1 day
- Cost: €499 (one-time, lifetime)
- Maintenance: $0/year
- **3-year total: €499**

**Savings: $1.4M-2.2M (99.97% cheaper)**

---

### vs Cloud (Together.ai)

**At low volume (<100M tokens/month):**
- Together.ai: $264/month = $3,168/year
- rbee electricity: $24,000/year
- **Together.ai cheaper**

**At high volume (300M+ tokens/month):**
- Together.ai: $792/month = $9,504/year
- rbee electricity: $24,000/year
- **Still Together.ai cheaper**

**But wait - at 500M+ tokens/month:**
- Together.ai: $1,320/month = $15,840/year
- rbee electricity: $24,000/year
- **Still not cheaper...**

**So when does rbee win on cost?**

**At 2B+ tokens/month:**
- Together.ai: $5,280/month = $63,360/year
- rbee electricity: $24,000/year
- **rbee saves: $39,360/year**

**BUT - Privacy & Control (Priceless):**
- ✅ rbee: Data NEVER leaves your network (required for healthcare, finance)
- ✅ rbee: Use ANY model (your fine-tuned models)
- ✅ rbee: Full GDPR control (your infrastructure)
- ✅ rbee: No vendor lock-in

**Real verdict:** Cloud for convenience + low volume. rbee for control + privacy + high volume.

---

## Example Business Model

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

## Real-World Business Scenarios

### Scenario 1: SaaS AI Platform 🐝

**Your business:** AI-powered writing assistant (like Jasper.ai)

**Your infrastructure:**
- 10x A100 GPUs (text generation)
- 1,000 customers (100 free, 800 pro, 100 enterprise)

**The colony setup:**

```rhai
// 🐝 Multi-tenant routing with customer tiers

fn route_task(task, workers) {
    // Enterprise customers get dedicated worker bees
    if task.customer_tier == "enterprise" {
        return workers
            .filter(|w| w.dedicated_to == task.customer_id)
            .first();
    }
    
    // Pro customers share worker bees (fair scheduling)
    if task.customer_tier == "pro" {
        return workers
            .filter(|w| w.tier == "pro")
            .least_loaded();
    }
    
    // Free tier gets specific worker bee only
    if task.customer_tier == "free" {
        return workers
            .filter(|w| w.hive == "gpu-node-01")
            .filter(|w| w.gpu_index == 0)
            .first();
    }
}

fn should_admit(task, customer) {
    // Free tier: 10K tokens/day
    if customer.tier == "free" && customer.tokens_used_today > 10000 {
        return reject("Daily limit reached. Upgrade to Pro for 1M tokens/day.");
    }
    
    // Pro tier: 1M tokens/day
    if customer.tier == "pro" && customer.tokens_used_today > 1000000 {
        return reject("Daily limit reached. Upgrade to Enterprise for unlimited.");
    }
    
    // Enterprise: unlimited
    return admit();
}
```

**What's happening:**
- 🐝 100 enterprise customers: Each gets dedicated worker bee
- 🐝 800 pro customers: Share pool of worker bees (fair scheduling)
- 🐝 100 free customers: Share single worker bee (rate limited)
- 🐝 Queen enforces quotas automatically
- 🐝 Your colony delivers: 1,000 customers, one platform

**Your revenue:**
- Free: $0 × 100 = $0
- Pro: $99/month × 800 = $79,200/month
- Enterprise: $999/month × 100 = $99,900/month
- **Total: $179,100/month = $2.15M/year**

**Your costs:**
- 10x A100 electricity: ~$2,000/month
- rbee premium (complete bundle): €499 (one-time)
- **Total Year 1: $24,499**

**Your profit: $2.12M/year (98.9% margin)**

---

### Scenario 2: EU-Compliant AI Service (Healthcare) 🐝

**Your business:** Healthcare AI provider (GDPR mandatory)

**Your infrastructure:**
- 5x H100 GPUs in Frankfurt (EU)
- 5x H100 GPUs in Virginia (US)
- Healthcare customers MUST use EU workers only

**The colony setup:**

```rhai
// 🐝 GDPR-compliant routing (EU-only for healthcare)

fn route_task(task, workers) {
    // Healthcare customers MUST use EU worker bees
    if task.customer_industry == "healthcare" {
        let eu_workers = workers.filter(|w| w.region == "EU");
        
        if eu_workers.is_empty() {
            return reject("No EU workers available. GDPR requirement.");
        }
        
        return eu_workers.least_loaded();
    }
    
    // US customers can use US worker bees
    if task.customer_region == "US" {
        return workers
            .filter(|w| w.region == "US")
            .least_loaded();
    }
    
    // Default: prefer same region as customer
    return workers
        .filter(|w| w.region == task.customer_region)
        .least_loaded();
}

fn should_admit(task, customer) {
    // Healthcare customers MUST have GDPR consent
    if customer.industry == "healthcare" && !customer.gdpr_consent {
        return reject("GDPR consent required for healthcare data.");
    }
    
    return admit();
}
```

**GDPR features (Premium GDPR Auditing €249):**

```bash
# Automatic audit trail (immutable, 7-year retention)
# ~/.rbee/audit/2025-11-03.log

{
  "timestamp": "2025-11-03T14:30:00Z",
  "customer_id": "healthcare-customer-123",
  "customer_industry": "healthcare",
  "api_key_fingerprint": "sha256:a3f2...",
  "endpoint": "/v1/chat/completions",
  "model": "llama-3-70b-medical",
  "tokens_in": 250,
  "tokens_out": 800,
  "worker_used": "eu-frankfurt-gpu-01:0",
  "worker_region": "EU",
  "duration_ms": 3200,
  "ip_address": "203.0.113.42",
  "gdpr_consent": true,
  "data_classification": "sensitive-health",
  "audit_hash": "sha256:b4e1..."  // Cryptographic integrity
}

# Article 15: Right to access
curl https://api.yourcompany.com/gdpr/export \
  -H "Authorization: Bearer healthcare-customer-123"
# Returns: All data for this customer (JSON export)

# Article 17: Right to erasure
curl -X DELETE https://api.yourcompany.com/gdpr/delete \
  -H "Authorization: Bearer healthcare-customer-123"
# Deletes: All customer data (audit logs preserved for 7 years)

# Article 20: Right to data portability
curl https://api.yourcompany.com/gdpr/export?format=json \
  -H "Authorization: Bearer healthcare-customer-123"
# Returns: Machine-readable JSON export
```

**What's happening:**
- 🐝 Healthcare customers ALWAYS routed to EU worker bees
- 🐝 US customers can use US worker bees (lower latency)
- 🐝 Every API call logged immutably (GDPR audit trail)
- 🐝 Automatic GDPR endpoints (export, delete, portability)
- 🐝 Your colony: GDPR-compliant Day 1

**Result:** Avoid €20M GDPR fines, serve EU healthcare market

---

### Scenario 3: Multi-Modal Content Platform 🐝

**Your business:** AI content generation platform (text + images + audio + video)

**Your infrastructure:**
- 10x A100 GPUs (text, audio)
- 5x RTX 4090 GPUs (images - cost-effective)
- 5x H100 GPUs (video - high performance)

**The colony setup:**

```rhai
// 🐝 Multi-modal routing (optimize cost vs performance)

fn route_task(task, workers) {
    match task.type {
        // Text: Use A100 worker bees (balanced)
        "text-gen" => workers
            .filter(|w| w.gpu_type == "A100")
            .least_loaded(),
        
        // Images: Use RTX 4090 worker bees (cost-effective)
        "image-gen" => workers
            .filter(|w| w.gpu_type == "RTX4090")
            .least_loaded(),
        
        // Audio: Use A100 worker bees (good for audio)
        "audio-gen" => workers
            .filter(|w| w.gpu_type == "A100")
            .filter(|w| w.capability == "audio")
            .least_loaded(),
        
        // Video: Use H100 worker bees (high performance required)
        "video-gen" => workers
            .filter(|w| w.gpu_type == "H100")
            .least_loaded(),
        
        _ => workers.least_loaded()
    }
}

fn calculate_cost(task, worker) {
    // Cost per GPU type (your internal accounting)
    let gpu_cost = match worker.gpu_type {
        "H100" => 0.10,      // $0.10/minute
        "A100" => 0.05,      // $0.05/minute
        "RTX4090" => 0.02,   // $0.02/minute
        _ => 0.01
    };
    
    return gpu_cost * task.estimated_duration_minutes;
}
```

**Your pricing tiers:**

| Tier | Monthly Price | Text | Images | Audio | Video |
|------|---------------|------|--------|-------|-------|
| **Starter** | $29 | 100K tokens | 50 images | 10 minutes | - |
| **Pro** | $99 | 1M tokens | 500 images | 100 minutes | 10 minutes |
| **Business** | $499 | 10M tokens | 5,000 images | 1,000 minutes | 100 minutes |
| **Enterprise** | Custom | Unlimited | Unlimited | Unlimited | Unlimited |

**What's happening:**
- 🐝 Text requests → A100 worker bees (balanced cost/performance)
- 🐝 Image requests → RTX 4090 worker bees (cost-effective)
- 🐝 Audio requests → A100 worker bees (good for audio models)
- 🐝 Video requests → H100 worker bees (high performance)
- 🐝 Queen optimizes cost automatically

**Your costs (monthly):**
- 10x A100: ~$2,000 electricity
- 5x RTX 4090: ~$500 electricity
- 5x H100: ~$1,500 electricity
- **Total: ~$4,000/month**

**Your revenue (1,000 customers):**
- 200 Starter × $29 = $5,800
- 600 Pro × $99 = $59,400
- 180 Business × $499 = $89,820
- 20 Enterprise × $2,000 = $40,000
- **Total: $195,020/month = $2.34M/year**

**Your profit: $2.29M/year (98.3% margin)**

---

## Advanced Features

### Custom Model Catalog 🐝

**Offer your fine-tuned models:**

```bash
# Add your custom models to the colony
rbee model add your-medical-llm-v2 --hive eu-frankfurt-01
rbee model add your-legal-llm-v3 --hive us-virginia-01
rbee model add your-custom-sdxl --hive gpu-node-05

# Customers access them via API
curl https://api.yourcompany.com/v1/models
# {
#   "models": [
#     {"id": "your-medical-llm-v2", "type": "text", "region": "EU"},
#     {"id": "your-legal-llm-v3", "type": "text", "region": "US"},
#     {"id": "your-custom-sdxl", "type": "image"}
#   ]
# }

# Customers use your models
curl https://api.yourcompany.com/v1/chat/completions \
  -H "Authorization: Bearer customer-key" \
  -d '{
    "model": "your-medical-llm-v2",
    "messages": [...]
  }'
```

**Result:** Monetize your fine-tuned models, differentiate from competitors

---

### Usage Tracking & Billing 🐝

**Automatic usage tracking:**

```bash
# Query customer usage (for billing)
curl https://api.yourcompany.com/admin/usage/customer-123?month=2025-11
# {
#   "customer_id": "customer-123",
#   "month": "2025-11",
#   "usage": {
#     "text_tokens": 1250000,
#     "images_generated": 450,
#     "audio_minutes": 85,
#     "video_minutes": 12
#   },
#   "costs": {
#     "text": 62.50,
#     "images": 9.00,
#     "audio": 4.25,
#     "video": 12.00,
#     "total": 87.75
#   },
#   "tier": "pro",
#   "overage": 0.00
# }

# Export for billing system
curl https://api.yourcompany.com/admin/usage/export?month=2025-11 \
  > billing_2025_11.csv
```

**Result:** Automatic usage tracking, easy billing integration

---

## Getting Started

### Prerequisites

- GPU infrastructure (dedicated servers or cloud VMs)
- SSH access between machines
- Basic command-line knowledge

### Installation

```bash
# 1. Install rbee
cargo install rbee-keeper

# 2. Configure hives
nano ~/.config/rbee/hives.conf

# 3. Install hives
rbee hive install <hostname>

# 4. Start queen
rbee queen start

# 5. Test your colony
curl http://localhost:7833/v1/models
```

**Full guide:** See main [README.md](../../README.md)

---

## Next Steps

1. **Evaluate:** Read [Competitive Analysis](04_COMPETITIVE_ANALYSIS.md)
2. **Premium:** Review [Premium Products](05_PREMIUM_PRODUCTS.md)
3. **Timeline:** Check [Implementation Roadmap](06_IMPLEMENTATION_ROADMAP.md)
4. **Deploy:** See main [README.md](../../README.md)

---

**🐝 Turn your GPU farm into a thriving hive in one day\!**
