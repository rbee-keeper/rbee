# TEAM-409: Automated Update Strategy & Cost Analysis

**Date:** 2025-11-05  
**Question:** What's the most cost-effective update interval for model lists?

---

## 💰 TL;DR: Update Strategy

**Recommended:**
- **Top 100 list:** Update every 24 hours (GitHub Actions)
- **Individual pages:** Update every 48 hours (ISR)
- **Total cost:** $0/month (all free tier)

---

## 🔄 Update Strategies

### Strategy 1: Scheduled Rebuilds (GitHub Actions) ✅ RECOMMENDED

**How it works:**
```yaml
# .github/workflows/update-models.yml
name: Update Model List

on:
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight UTC
  workflow_dispatch:      # Manual trigger

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Fetch latest top 100
        run: npm run fetch:top100
      
      - name: Generate static pages
        run: npm run generate:ssg
      
      - name: Deploy to Cloudflare Pages
        run: wrangler pages deploy dist/
```

**Cost:** $0/month (GitHub Actions free tier: 2,000 minutes/month)

---

### Strategy 2: Cloudflare Cron Triggers ✅ ALSO RECOMMENDED

**How it works:**
```typescript
// Cloudflare Worker with Cron
export default {
  async scheduled(event, env, ctx) {
    // Runs every 24 hours
    await updateTop100Models(env);
  },
  
  async fetch(request, env) {
    // Serve cached models
  }
};

// wrangler.toml
[triggers]
crons = ["0 0 * * *"]  # Daily at midnight
```

**Cost:** $0/month (Cloudflare Cron is free)

---

### Strategy 3: ISR with Time-based Revalidation ✅ BEST FOR INDIVIDUAL PAGES

**How it works:**
```typescript
// Next.js ISR
export async function getStaticProps() {
  const model = await fetchModel();
  
  return {
    props: { model },
    revalidate: 172800, // 48 hours in seconds
  };
}
```

**Cost:** $0/month (revalidation happens on-demand)

---

## 📊 Cost Analysis: Different Update Intervals

### Scenario: Top 100 Model List

| Interval | Updates/Month | GitHub Actions | Cloudflare | Cost |
|----------|---------------|----------------|------------|------|
| **1 hour** | 720 | 72 minutes | Free | $0 |
| **6 hours** | 120 | 12 minutes | Free | $0 |
| **12 hours** | 60 | 6 minutes | Free | $0 |
| **24 hours** | 30 | 3 minutes | Free | $0 ✅ |
| **48 hours** | 15 | 1.5 minutes | Free | $0 |
| **Weekly** | 4 | 0.4 minutes | Free | $0 |

**All intervals are FREE!** GitHub Actions gives 2,000 minutes/month.

---

### Scenario: Individual Model Pages (40K pages)

| Interval | Strategy | Requests/Month | Cost |
|----------|----------|----------------|------|
| **Real-time** | Workers | Unlimited | $5/month |
| **1 hour** | ISR | ~960K | $0 (Cloudflare) |
| **6 hours** | ISR | ~160K | $0 |
| **24 hours** | ISR | ~40K | $0 ✅ |
| **48 hours** | ISR | ~20K | $0 ✅ |
| **Weekly** | ISR | ~5.7K | $0 |

**All intervals are FREE with ISR!**

---

## 🎯 Recommended Update Intervals

### 1. Top 100 Model List: **24 hours** ✅

**Why:**
- HuggingFace rankings change daily
- Captures new trending models
- Still very fresh for users
- Minimal compute cost

**Implementation:**
```yaml
# .github/workflows/update-top100.yml
on:
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight UTC
```

**Cost:** $0/month  
**Freshness:** 24 hours max

---

### 2. Individual Model Pages: **48 hours** ✅

**Why:**
- Model metadata rarely changes
- 48 hours is fresh enough
- Reduces unnecessary API calls
- Lower cache churn

**Implementation:**
```typescript
export async function getStaticProps() {
  return {
    props: { model },
    revalidate: 172800, // 48 hours
  };
}
```

**Cost:** $0/month  
**Freshness:** 48 hours max

---

### 3. Model Compatibility Check: **On-demand** ✅

**Why:**
- Compatibility logic is deterministic
- No need to recheck unless model changes
- Can be cached forever

**Implementation:**
```typescript
// Cache compatibility result forever
const cacheKey = `compat:${modelId}`;
const cached = await env.KV.get(cacheKey);
if (cached) return cached;

const result = isModelCompatible(metadata);
await env.KV.put(cacheKey, result); // No expiry
```

**Cost:** $0/month  
**Freshness:** Instant

---

## 🔧 Implementation Architecture

### Option 1: GitHub Actions + Cloudflare Pages (SIMPLEST) ✅

```
┌─────────────────────────────────────────────┐
│ GitHub Actions (Cron: Daily)                │
│ ├─ Fetch top 100 from HuggingFace          │
│ ├─ Filter compatible models                │
│ ├─ Generate static pages                   │
│ └─ Deploy to Cloudflare Pages              │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ Cloudflare Pages (Static Hosting)          │
│ ├─ /models (list page - updated daily)     │
│ ├─ /models/[id] (ISR - 48h revalidation)   │
│ └─ Edge caching (global CDN)               │
└─────────────────────────────────────────────┘
```

**Cost:** $0/month  
**Complexity:** Low  
**Freshness:** 24h (list), 48h (pages)

---

### Option 2: Cloudflare Workers + Cron + KV (MOST FLEXIBLE) ✅

```
┌─────────────────────────────────────────────┐
│ Cloudflare Cron Trigger (Daily)            │
│ ├─ Fetch top 100 from HuggingFace          │
│ ├─ Filter compatible models                │
│ ├─ Store in KV with 24h TTL                │
│ └─ Trigger cache purge                     │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ Cloudflare Worker (Edge Runtime)           │
│ ├─ Check KV for cached list                │
│ ├─ Render model pages dynamically          │
│ ├─ Cache with 48h TTL                      │
│ └─ Global edge caching                     │
└─────────────────────────────────────────────┘
```

**Cost:** $5/month  
**Complexity:** Medium  
**Freshness:** Real-time capable

---

## 💡 Smart Update Strategy

### Tiered Update Frequency

| Content Type | Update Interval | Reason |
|-------------|-----------------|---------|
| **Top 10 models** | 6 hours | Most popular, changes fast |
| **Top 100 models** | 24 hours | Popular, moderate changes |
| **Top 1K models** | 48 hours | Less popular, slow changes |
| **Long tail (39K)** | On-demand | Rarely accessed |

**Implementation:**
```typescript
function getRevalidationTime(rank: number): number {
  if (rank <= 10) return 21600;    // 6 hours
  if (rank <= 100) return 86400;   // 24 hours
  if (rank <= 1000) return 172800; // 48 hours
  return false; // Never revalidate (on-demand only)
}
```

---

## 📊 Cost Analysis: Update Frequency

### GitHub Actions Cost (Free Tier: 2,000 min/month)

| Update Interval | Minutes/Month | % of Free Tier | Cost |
|-----------------|---------------|----------------|------|
| **Hourly** | 720 min | 36% | $0 |
| **Every 6h** | 120 min | 6% | $0 |
| **Daily** | 30 min | 1.5% | $0 ✅ |
| **Every 48h** | 15 min | 0.75% | $0 |

**All FREE!** Even hourly updates fit in free tier.

---

### Cloudflare Workers Cost

| Requests/Month | Free Tier | Paid Tier | Cost |
|----------------|-----------|-----------|------|
| **100K** | ✅ Free | - | $0 |
| **1M** | ❌ | $5/month | $5 |
| **10M** | ❌ | $5/month | $5 |

**Note:** Cloudflare Workers pricing is FLAT at $5/month after free tier.

---

### Cloudflare KV Cost

| Storage | Reads/Month | Writes/Month | Cost |
|---------|-------------|--------------|------|
| **1MB** | 100K | 1K | $0 |
| **10MB** | 1M | 10K | $0.50 |
| **100MB** | 10M | 100K | $5.50 |

**For our use case (top 100 updates):**
- Storage: ~1MB
- Reads: ~100K/month
- Writes: ~100/month (daily updates)
- **Cost: $0/month** ✅

---

## 🎯 Recommended Implementation

### Phase 1: GitHub Actions + Static (Week 1)

```yaml
# .github/workflows/update-models.yml
name: Update Model List

on:
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight UTC
  workflow_dispatch:

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node
        uses: actions/setup-node@v3
        with:
          node-version: '20'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Fetch top 100 models
        run: npm run fetch:top100
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
      
      - name: Generate static pages
        run: npm run build
      
      - name: Deploy to Cloudflare Pages
        run: npx wrangler pages deploy dist/
        env:
          CLOUDFLARE_API_TOKEN: ${{ secrets.CF_API_TOKEN }}
```

**Cost:** $0/month  
**Complexity:** Low  
**Setup time:** 1 hour

---

### Phase 2: Add ISR for Individual Pages (Week 2)

```typescript
// app/models/[id]/page.tsx
export const revalidate = 172800; // 48 hours

export async function generateStaticParams() {
  const top100 = await getTop100Models();
  return top100.map(m => ({ id: m.id }));
}

export default async function ModelPage({ params }) {
  const model = await fetchModel(params.id);
  const compat = isModelCompatible(model.metadata);
  
  return <ModelDetails model={model} compat={compat} />;
}
```

**Cost:** $0/month  
**Complexity:** Low  
**Setup time:** 2 hours

---

### Phase 3: Cloudflare Workers for Real-time (Month 2)

```typescript
// worker.ts
export default {
  // Cron trigger: Update top 100 daily
  async scheduled(event, env, ctx) {
    const models = await fetchTop100FromHF();
    const compatible = models.filter(m => 
      isModelCompatible(m.metadata).compatible
    );
    
    // Store in KV with 24h TTL
    await env.MODELS.put('top100', JSON.stringify(compatible), {
      expirationTtl: 86400, // 24 hours
    });
  },
  
  // HTTP handler: Serve models
  async fetch(request, env) {
    const url = new URL(request.url);
    
    if (url.pathname === '/models') {
      // Serve top 100 list
      const cached = await env.MODELS.get('top100');
      return Response.json(JSON.parse(cached));
    }
    
    // Serve individual model page
    const modelId = url.pathname.split('/')[2];
    const cached = await env.MODELS.get(`model:${modelId}`);
    
    if (cached) {
      return new Response(cached, {
        headers: { 'Content-Type': 'text/html' },
      });
    }
    
    // Generate and cache
    const html = await generateModelPage(modelId);
    await env.MODELS.put(`model:${modelId}`, html, {
      expirationTtl: 172800, // 48 hours
    });
    
    return new Response(html);
  }
};
```

**Cost:** $5/month  
**Complexity:** Medium  
**Setup time:** 4 hours

---

## 📈 Cost-Effectiveness Analysis

### Update Frequency vs Cost

| Interval | GitHub Actions | Cloudflare | Total | Freshness |
|----------|----------------|------------|-------|-----------|
| **Real-time** | N/A | $5/month | $5 | Instant |
| **1 hour** | $0 | $0 | $0 | 1h max |
| **6 hours** | $0 | $0 | $0 | 6h max |
| **24 hours** | $0 | $0 | $0 ✅ | 24h max |
| **48 hours** | $0 | $0 | $0 ✅ | 48h max |
| **Weekly** | $0 | $0 | $0 | 7d max |

**Most cost-effective: 24-48 hours** ✅

---

## 🎯 Final Recommendations

### For Top 100 Model List:
- **Update interval:** 24 hours
- **Method:** GitHub Actions cron
- **Cost:** $0/month
- **Reason:** Daily changes in rankings, free tier sufficient

### For Individual Model Pages:
- **Update interval:** 48 hours
- **Method:** ISR (Incremental Static Regeneration)
- **Cost:** $0/month
- **Reason:** Metadata rarely changes, reduces API calls

### For Model Compatibility:
- **Update interval:** On-demand (cache forever)
- **Method:** Cloudflare KV
- **Cost:** $0/month
- **Reason:** Deterministic logic, no need to recheck

---

## ✅ Implementation Checklist

- [ ] Set up GitHub Actions workflow
- [ ] Configure daily cron trigger (0 0 * * *)
- [ ] Add HuggingFace API token to secrets
- [ ] Configure Cloudflare Pages deployment
- [ ] Set up ISR with 48h revalidation
- [ ] Add monitoring for failed updates
- [ ] Set up alerts for stale data

---

## 📊 Expected Costs

### Conservative Estimate
- GitHub Actions: $0/month (free tier)
- Cloudflare Pages: $0/month (free tier)
- Cloudflare KV: $0/month (free tier)
- **Total: $0/month** ✅

### With Workers (if needed)
- GitHub Actions: $0/month
- Cloudflare Workers: $5/month
- Cloudflare KV: $0.50/month
- **Total: $5.50/month** ✅

---

## 💡 Pro Tips

### 1. Cache HuggingFace API Responses
```typescript
// Cache API responses to reduce rate limits
const cacheKey = `hf:top100:${date}`;
const cached = await redis.get(cacheKey);
if (cached) return JSON.parse(cached);

const models = await fetchFromHF();
await redis.set(cacheKey, JSON.stringify(models), 'EX', 86400);
```

### 2. Use Conditional Requests
```typescript
// Only fetch if data changed (saves bandwidth)
const etag = await env.KV.get('top100:etag');
const response = await fetch(HF_API, {
  headers: { 'If-None-Match': etag },
});

if (response.status === 304) {
  // Data unchanged, skip update
  return;
}
```

### 3. Batch Updates
```typescript
// Update multiple pages in one workflow run
const models = await getTop100();
await Promise.all(
  models.map(m => generateModelPage(m))
);
```

---

## 📝 Summary

**Question:** What's the most cost-effective update interval?

**Answer:**
- **Top 100 list:** 24 hours (GitHub Actions)
- **Individual pages:** 48 hours (ISR)
- **Compatibility:** On-demand (cache forever)
- **Total cost:** $0/month

**Why this works:**
- ✅ GitHub Actions free tier: 2,000 min/month
- ✅ Cloudflare Pages: Unlimited bandwidth
- ✅ ISR: On-demand revalidation
- ✅ All updates fit in free tiers

**This strategy gives you fresh data at ZERO cost!** 🎉

---

**TEAM-409 - Update Strategy Complete** ✅  
**Recommendation: 24h list updates, 48h page updates, $0/month** 🚀
