# rbee: Business Use Case

**Audience:** GPU infrastructure operators, AI service providers, startups  
**Date:** November 2, 2025

---

## The Problem: "Turn Your GPU Farm Into a Product"

### Your Situation

You're a business with GPU infrastructure:
- **20x NVIDIA A100 GPUs** (80GB each) - $1.6M investment
- **10x NVIDIA H100 GPUs** (80GB each) - $3M investment
- **Mix of capabilities:** Text, images, audio, video
- **Goal:** Offer AI services to customers and monetize your infrastructure

### Current Reality: Platform Complexity Hell

**You want to offer AI services, but:**

❌ **Text + Images + Audio = 3 Different Platforms**
- vLLM for text generation
- ComfyUI for image generation
- Whisper for audio transcription
- Each has different APIs, configs, monitoring

❌ **Customer Fragmentation**
- Customers need different API keys for each service
- Different billing systems for each platform
- Different SLA tracking for each service

❌ **No Control Over Resource Allocation**
- Can't easily route enterprise customers to H100s
- Can't enforce quotas per customer
- Can't prioritize by customer tier

❌ **No Built-in Compliance**
- GDPR compliance = custom implementation
- Audit logging = custom implementation
- Data residency = manual enforcement

❌ **Scaling Nightmare**
- Adding new modality = new platform
- Managing 3+ separate systems
- 6-12 months development per platform

**Total cost to build from scratch: $500K+ in engineering**

---

## The rbee Solution: One Platform, All Modalities

### Setup (One Day)

```bash
# 1. Install rbee on control node
cargo install rbee-keeper

# 2. Configure GPU nodes
cat > ~/.config/rbee/hives.conf << 'EOF'
Host gpu-node-01
  HostName 10.0.1.101
  User rbee
  HivePort 7835
  # 4x A100 GPUs

Host gpu-node-02
  HostName 10.0.1.102
  User rbee
  HivePort 7835
  # 4x A100 GPUs

Host gpu-node-03
  HostName 10.0.1.103
  User rbee
  HivePort 7835
  # 2x H100 GPUs

# ... repeat for all nodes
EOF

# 3. Install hives on all nodes (SSH-based, like Ansible)
rbee hive install gpu-node-01
rbee hive install gpu-node-02
rbee hive install gpu-node-03
# ... repeat for all nodes

# 4. Configure your model catalog
rbee model add llama-3-405b --hive gpu-node-03  # H100 for large models
rbee model add llama-3-70b --hive gpu-node-01   # A100 for medium models
rbee model add sdxl --hive gpu-node-01          # A100 for images
rbee model add whisper-large --hive gpu-node-02 # A100 for audio

# Done! ✅
```

### Configure Multi-Tenancy (Rhai Script)

```bash
cat > ~/.config/rbee/scheduler.rhai << 'EOF'
// Multi-tenant routing with quotas and tiers

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
    if task.customer_tier == "free" {
        return workers
            .filter(|w| w.hive == "gpu-node-01")
            .filter(|w| w.gpu_index == 0)  // Only first GPU
            .first();
    }
    
    // Route by task type
    match task.type {
        "image-gen" => workers
            .filter(|w| w.capability == "image-gen")
            .least_loaded(),
        "audio-gen" => workers
            .filter(|w| w.capability == "audio-gen")
            .least_loaded(),
        _ => workers.least_loaded()
    }
}

fn should_admit(task, customer) {
    // Check daily quota
    if customer.tokens_used_today > customer.daily_limit {
        return reject("Daily quota exceeded. Upgrade your plan.");
    }
    
    // Check tier access to models
    if task.model == "llama-3-405b" && customer.tier != "enterprise" {
        return reject("llama-3-405b requires Enterprise tier");
    }
    
    if task.model == "llama-3-70b" && customer.tier == "free" {
        return reject("llama-3-70b requires Pro tier or higher");
    }
    
    // Check concurrent request limit
    if customer.active_requests >= customer.max_concurrent {
        return reject("Concurrent request limit reached");
    }
    
    return admit();
}

fn calculate_priority(task, customer) {
    // Enterprise = highest priority
    if customer.tier == "enterprise" {
        return 100;
    }
    
    // Pro = medium priority
    if customer.tier == "pro" {
        return 50;
    }
    
    // Free = lowest priority
    return 10;
}
EOF
```

**Save and activate:**
```bash
rbee scheduler reload
# ✅ Multi-tenancy active, no code changes needed!
```

---

## What Your Customers See

### One OpenAI-Compatible API for Everything

```bash
# Your customers get ONE endpoint:
# https://api.yourcompany.com/v1/

# Text generation
curl https://api.yourcompany.com/v1/chat/completions \
  -H "Authorization: Bearer customer-key-123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-405b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Image generation
curl https://api.yourcompany.com/v1/images/generations \
  -H "Authorization: Bearer customer-key-123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sdxl",
    "prompt": "a cat wearing a top hat",
    "size": "1024x1024"
  }'

# Audio transcription
curl https://api.yourcompany.com/v1/audio/transcriptions \
  -H "Authorization: Bearer customer-key-123" \
  -F file=@audio.mp3 \
  -F model=whisper-large

# All from the SAME endpoint
# All with the SAME API key
# All billed together
```

---

## Business Features Out of the Box

### 1. Multi-Tenancy & Quotas

**Automatic enforcement via Rhai:**
```rhai
fn should_admit(task, customer) {
    // Daily token quota
    if customer.tokens_used_today > customer.daily_limit {
        return reject("Quota exceeded");
    }
    
    // Model access by tier
    if task.model == "llama-3-405b" && customer.tier != "enterprise" {
        return reject("Upgrade to Enterprise");
    }
    
    // Concurrent request limit
    if customer.active_requests >= customer.max_concurrent {
        return reject("Too many concurrent requests");
    }
    
    return admit();
}
```

**Customer tiers:**
- **Free:** 10K tokens/day, llama-3-8b only, 1 concurrent request
- **Pro:** 1M tokens/day, llama-3-70b, 10 concurrent requests
- **Enterprise:** Unlimited, llama-3-405b, 100 concurrent requests

---

### 2. GDPR Compliance (Built-in)

**Automatic audit logging:**
```bash
# Every API call is logged immutably
# ~/.rbee/audit.log (7-year retention)

{
  "timestamp": "2025-11-02T21:00:00Z",
  "customer_id": "customer-123",
  "api_key_fingerprint": "a3f2...",
  "endpoint": "/v1/chat/completions",
  "model": "llama-3-70b",
  "tokens_in": 150,
  "tokens_out": 500,
  "gpu_used": "gpu-node-01:0",
  "duration_ms": 2340,
  "ip_address": "203.0.113.42",
  "user_agent": "openai-python/1.0.0"
}
```

**GDPR endpoints (automatic):**
```bash
# Data export
curl https://api.yourcompany.com/gdpr/export \
  -H "Authorization: Bearer customer-key-123"
# Returns all data for this customer

# Data deletion
curl -X DELETE https://api.yourcompany.com/gdpr/delete \
  -H "Authorization: Bearer customer-key-123"
# Deletes all customer data (except audit logs)

# Consent tracking
curl https://api.yourcompany.com/gdpr/consent \
  -H "Authorization: Bearer customer-key-123" \
  -d '{"consent": true, "purpose": "ai-inference"}'
```

**EU-only routing:**
```rhai
fn route_task(task, workers) {
    // EU customers must use EU workers
    if task.customer_region == "EU" {
        return workers
            .filter(|w| w.region == "EU")
            .least_loaded();
    }
    
    return workers.least_loaded();
}
```

---

### 3. Cost Control & Optimization

**Route by cost:**
```rhai
fn route_task(task, workers) {
    // Small models → cheap GPUs (RTX 4090)
    if task.model == "llama-3-8b" {
        return workers
            .filter(|w| w.gpu_type == "RTX4090")
            .least_loaded();
    }
    
    // Medium models → mid-tier GPUs (A100)
    if task.model == "llama-3-70b" {
        return workers
            .filter(|w| w.gpu_type == "A100")
            .least_loaded();
    }
    
    // Large models → expensive GPUs (H100)
    if task.model == "llama-3-405b" {
        return workers
            .filter(|w| w.gpu_type == "H100")
            .least_loaded();
    }
    
    return workers.least_loaded();
}
```

**Time-based routing:**
```rhai
fn route_task(task, workers) {
    let hour = current_time().hour();
    
    // Peak hours (9am-5pm): use all GPUs
    if hour >= 9 && hour <= 17 {
        return workers.least_loaded();
    }
    
    // Off-peak: use cheaper GPUs only
    return workers
        .filter(|w| w.gpu_type != "H100")
        .least_loaded();
}
```

---

### 4. Custom Model Catalog

**Offer your fine-tuned models:**
```bash
# Add your custom models
rbee model add your-custom-llm-v2 \
  --hive gpu-node-01 \
  --path /models/custom-llm-v2.gguf

rbee model add your-custom-sdxl \
  --hive gpu-node-02 \
  --path /models/custom-sdxl.safetensors

# Customers see them in API
curl https://api.yourcompany.com/v1/models
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "llama-3-405b",
      "object": "model",
      "created": 1699000000,
      "owned_by": "meta",
      "permission": ["enterprise"]
    },
    {
      "id": "your-custom-llm-v2",
      "object": "model",
      "created": 1730000000,
      "owned_by": "yourcompany",
      "permission": ["pro", "enterprise"]
    },
    {
      "id": "your-custom-sdxl",
      "object": "model",
      "created": 1730000000,
      "owned_by": "yourcompany",
      "permission": ["all"]
    }
  ]
}
```

---

### 5. Monitoring & Observability

**Built-in metrics:**
```bash
# Prometheus metrics endpoint
curl http://localhost:7833/metrics
```

**Metrics exported:**
- `rbee_requests_total{customer_tier, model, status}`
- `rbee_tokens_total{customer_tier, model, direction}`
- `rbee_duration_seconds{customer_tier, model, percentile}`
- `rbee_gpu_utilization{hive, gpu_index, gpu_type}`
- `rbee_queue_depth{customer_tier}`
- `rbee_active_workers{hive, capability}`

**Grafana dashboard (included):**
- Customer usage by tier
- GPU utilization by node
- Request latency (p50, p95, p99)
- Token throughput
- Error rates
- Queue depth

---

## Real-World Business Scenarios

### Scenario 1: SaaS AI Platform

**Company:** AI writing assistant startup  
**Infrastructure:** 10x A100 GPUs  
**Customers:** 1,000 users (100 free, 800 pro, 100 enterprise)

**Setup:**
```rhai
fn route_task(task, workers) {
    // Enterprise: dedicated GPUs
    if task.customer_tier == "enterprise" {
        return workers
            .filter(|w| w.hive == "gpu-node-01")  // Dedicated node
            .least_loaded();
    }
    
    // Pro: shared GPUs
    if task.customer_tier == "pro" {
        return workers
            .filter(|w| w.hive.starts_with("gpu-node-0"))
            .filter(|w| w.hive != "gpu-node-01")  // Not dedicated
            .least_loaded();
    }
    
    // Free: specific GPU only
    return workers
        .filter(|w| w.hive == "gpu-node-05")
        .filter(|w| w.gpu_index == 0)
        .first();
}
```

**Result:**
- ✅ Enterprise customers get dedicated resources
- ✅ Pro customers share resources efficiently
- ✅ Free tier doesn't impact paid tiers
- ✅ One platform, three tiers

---

### Scenario 2: EU-Compliant AI Service

**Company:** Healthcare AI provider  
**Requirement:** GDPR compliance, EU-only data processing  
**Infrastructure:** 20x A100 in EU, 10x H100 in US

**Setup:**
```rhai
fn route_task(task, workers) {
    // EU customers MUST use EU workers
    if task.customer_region == "EU" {
        let eu_workers = workers.filter(|w| w.region == "EU");
        
        if eu_workers.is_empty() {
            return reject("No EU workers available");
        }
        
        return eu_workers.least_loaded();
    }
    
    // US customers can use any
    return workers.least_loaded();
}
```

**Result:**
- ✅ Automatic EU data residency enforcement
- ✅ Audit logs prove compliance
- ✅ GDPR endpoints built-in
- ✅ Pass SOC2 audits

---

### Scenario 3: Multi-Modal Content Platform

**Company:** AI content generation platform  
**Services:** Text, images, audio, video  
**Infrastructure:** Mixed GPUs (A100, H100, RTX 4090)

**Setup:**
```rhai
fn route_task(task, workers) {
    match task.type {
        // Images: RTX 4090 (cost-effective)
        "image-gen" => workers
            .filter(|w| w.gpu_type == "RTX4090")
            .least_loaded(),
        
        // Video: H100 (high performance)
        "video-gen" => workers
            .filter(|w| w.gpu_type == "H100")
            .least_loaded(),
        
        // Text: A100 (balanced)
        "text-gen" => workers
            .filter(|w| w.gpu_type == "A100")
            .least_loaded(),
        
        // Audio: A100 (balanced)
        "audio-gen" => workers
            .filter(|w| w.gpu_type == "A100")
            .least_loaded(),
        
        _ => workers.least_loaded()
    }
}
```

**Result:**
- ✅ One API for all modalities
- ✅ Optimal GPU allocation per task type
- ✅ Cost-effective resource usage
- ✅ Customers get unified billing

---

## ROI Analysis

### Building from Scratch

**Development costs:**
- 3-5 senior engineers × 6-12 months
- Salary: $150K-200K/year per engineer
- **Total: $450K-1M in year 1**

**Ongoing costs:**
- Maintenance: 2 engineers
- **Total: $300K-400K/year**

**Time to market:** 6-12 months

---

### Using rbee (Self-Hosted)

**Setup costs:**
- 1 engineer × 1 day
- **Total: ~$500**

**Ongoing costs:**
- Community support: Free
- Updates: Automatic
- **Total: $0/year**

**Time to market:** 1 day

**Savings: $450K-1M in year 1**

---

### Using rbee (Managed Platform - Future)

**Setup costs:**
- Configuration: 1 hour
- **Total: ~$100**

**Ongoing costs:**
- Platform fee: 30-40% of revenue
- Enterprise support: Included
- SLA guarantees: Included

**Time to market:** 1 hour

**Trade-off:** Pay platform fee, get enterprise support + SLA

---

## Pricing Tiers (Example)

### Your Customer Pricing

| Tier | Monthly Price | Features |
|------|---------------|----------|
| **Free** | $0 | 10K tokens/day, llama-3-8b, 1 concurrent |
| **Pro** | $99 | 1M tokens/day, llama-3-70b, 10 concurrent |
| **Enterprise** | $999 | Unlimited, llama-3-405b, 100 concurrent, SLA |

### Your Costs (rbee Self-Hosted)

| Cost | Amount |
|------|--------|
| **Software** | $0 (GPL-3.0) |
| **GPU Infrastructure** | Already owned |
| **Electricity** | ~$0.50-1.00/GPU/hour |
| **Bandwidth** | Minimal (API responses) |

### Your Margins

**Example: 100 Pro customers**
- Revenue: $99 × 100 = $9,900/month
- GPU costs: ~$2,000/month (electricity)
- **Profit: ~$7,900/month = $94,800/year**

**Example: 10 Enterprise customers**
- Revenue: $999 × 10 = $9,990/month
- GPU costs: ~$3,000/month (dedicated resources)
- **Profit: ~$6,990/month = $83,880/year**

---

## Getting Started

### Prerequisites

- GPU infrastructure (A100, H100, or consumer GPUs)
- SSH access to all nodes
- Basic Rust knowledge (for Rhai scripting)

### Setup Steps

1. **Install rbee on control node**
   ```bash
   cargo install rbee-keeper
   ```

2. **Configure hives**
   ```bash
   nano ~/.config/rbee/hives.conf
   ```

3. **Install hives on GPU nodes**
   ```bash
   rbee hive install <hostname>
   ```

4. **Configure scheduler**
   ```bash
   nano ~/.config/rbee/scheduler.rhai
   ```

5. **Start queen**
   ```bash
   rbee queen start --features local-hive
   ```

6. **Test API**
   ```bash
   curl http://localhost:7833/v1/models
   ```

**Full guide:** See main [README.md](../../README.md)

---

## FAQ

**Q: Can rbee handle 1000+ customers?**  
A: Yes! rbee is designed for multi-tenancy. Rhai scheduler handles routing, quotas, and priorities.

**Q: Is rbee GDPR compliant?**  
A: Yes! Immutable audit logs, data export/deletion endpoints, and EU-only routing are built-in.

**Q: Can I use my own fine-tuned models?**  
A: Yes! Add any GGUF or SafeTensors model to your catalog.

**Q: What about billing integration?**  
A: rbee provides usage metrics. Integrate with Stripe/Chargebee for billing (custom implementation).

**Q: Do I need to write Rhai scripts?**  
A: Recommended Rhai templates are provided. Customize as needed.

**Q: What's the performance overhead?**  
A: Minimal. Rhai scripts compile to bytecode. Routing adds <1ms latency.

**Q: Can I run rbee in Kubernetes?**  
A: Yes! rbee binaries are containerizable. Helm charts coming in M2.

---

## Next Steps

1. **Evaluate:** Compare with [Technical Differentiators](04_TECHNICAL_DIFFERENTIATORS.md)
2. **Plan:** Review [Implementation Roadmap](06_IMPLEMENTATION_ROADMAP.md)
3. **Try:** See [README.md](../../README.md) for setup
4. **Contact:** Enterprise support (coming soon)

---

**Turn your GPU farm into a product in one day.** 🐝
