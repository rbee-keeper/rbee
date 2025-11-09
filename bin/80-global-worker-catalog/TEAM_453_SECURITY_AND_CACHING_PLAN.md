# TEAM-453: Worker Catalog Security & Caching Plan

**Date:** 2025-11-09  
**Status:** 📋 PLAN - Ready for Implementation

## Summary

The worker catalog (gwc.rbee.dev) needs:
1. **Rate limiting** - Prevent abuse
2. **Caching** - Reduce load and avoid rate limits on external APIs
3. **Security hardening** - CORS, headers, validation

## Current State

### ✅ What We Have
- CORS middleware (localhost origins only)
- Basic error handling
- Cache headers on responses (5 min for install.sh)

### ❌ What's Missing
- **No rate limiting** - Can be hammered
- **No caching layer** - Hits external APIs every time
- **CORS too restrictive** - Doesn't allow marketplace.rbee.dev
- **No request validation** - Accepts any input
- **No security headers** - Missing CSP, HSTS, etc.

## 1. Rate Limiting

### Why We Need It
- Prevent abuse/DoS attacks
- Protect Cloudflare Worker quotas
- Fair usage across clients

### Implementation

**Use Cloudflare Rate Limiting (Built-in)**

```typescript
// src/middleware/rateLimit.ts
import { Hono } from 'hono'

export function rateLimit() {
  return async (c, next) => {
    // Cloudflare provides rate limiting via:
    // 1. Wrangler config (wrangler.jsonc)
    // 2. Cloudflare Dashboard
    // 3. Or manual implementation with KV/Durable Objects
    
    // For now, use simple in-memory (resets on deploy)
    const ip = c.req.header('CF-Connecting-IP') || 'unknown'
    
    // TODO: Use Cloudflare KV for persistent rate limiting
    // For MVP: Trust Cloudflare's built-in protections
    
    await next()
  }
}
```

**Cloudflare Dashboard Rate Limiting:**
1. Go to Cloudflare Dashboard
2. Workers & Pages → gwc-rbee → Settings
3. Add Rate Limiting Rule:
   - **Requests:** 100 per minute per IP
   - **Action:** Block for 1 minute

### Rate Limits (Recommended)

| Endpoint | Limit | Window | Action |
|----------|-------|--------|--------|
| `/workers` | 60 req/min | Per IP | Block 1min |
| `/workers/:id` | 120 req/min | Per IP | Block 1min |
| `/workers/:id/PKGBUILD` | 30 req/min | Per IP | Block 5min |
| `/install.sh` | 10 req/min | Per IP | Block 10min |

## 2. Caching Strategy

### Problem: External API Rate Limits

**HuggingFace API:**
- Rate limit: ~100 requests/hour (unauthenticated)
- Marketplace SSG build makes 10-20 requests
- Risk: Build failures during deployment

**CivitAI API:**
- Rate limit: ~1000 requests/hour
- Less of a concern but still should cache

### Solution: Multi-Layer Caching

#### Layer 1: Cloudflare CDN Cache (Free)

```typescript
// src/routes.ts
routes.get('/workers', (c) => {
  return c.json(
    { workers: WORKERS },
    200,
    {
      'Cache-Control': 'public, max-age=3600, s-maxage=3600', // 1 hour
      'CDN-Cache-Control': 'max-age=3600',
      'Cloudflare-CDN-Cache-Control': 'max-age=3600',
    }
  )
})

routes.get('/workers/:id', (c) => {
  const id = c.req.param('id')
  const worker = WORKERS.find(w => w.id === id)
  
  if (!worker) {
    return c.json({ error: 'Worker not found' }, 404)
  }
  
  return c.json(
    worker,
    200,
    {
      'Cache-Control': 'public, max-age=3600, s-maxage=3600', // 1 hour
    }
  )
})
```

#### Layer 2: marketplace-node Caching

**For HuggingFace/CivitAI API calls:**

```typescript
// bin/79_marketplace_core/marketplace-node/src/cache.ts
import { writeFile, readFile, stat } from 'fs/promises'
import { join } from 'path'

const CACHE_DIR = '.cache/marketplace'
const CACHE_TTL = 3600 * 1000 // 1 hour

interface CacheEntry<T> {
  data: T
  timestamp: number
}

export async function getCached<T>(
  key: string,
  fetcher: () => Promise<T>,
  ttl: number = CACHE_TTL
): Promise<T> {
  const cacheFile = join(CACHE_DIR, `${key}.json`)
  
  try {
    // Check if cache exists and is fresh
    const stats = await stat(cacheFile)
    const age = Date.now() - stats.mtimeMs
    
    if (age < ttl) {
      const cached = JSON.parse(await readFile(cacheFile, 'utf-8')) as CacheEntry<T>
      console.log(`[cache] HIT ${key} (age: ${Math.round(age / 1000)}s)`)
      return cached.data
    }
  } catch (error) {
    // Cache miss or error - fetch fresh data
  }
  
  console.log(`[cache] MISS ${key} - fetching fresh data`)
  const data = await fetcher()
  
  // Save to cache
  await writeFile(
    cacheFile,
    JSON.stringify({ data, timestamp: Date.now() } as CacheEntry<T>)
  )
  
  return data
}
```

**Usage:**

```typescript
// bin/79_marketplace_core/marketplace-node/src/huggingface.ts
import { getCached } from './cache'

export async function fetchHFModels(
  query: string | undefined,
  options: { limit?: number; sort?: string } = {}
): Promise<HFModel[]> {
  const cacheKey = `hf-${query || 'all'}-${options.limit}-${options.sort}`
  
  return getCached(cacheKey, async () => {
    const params = new URLSearchParams({
      ...(query && { search: query }),
      limit: String(options.limit || 50),
      sort: options.sort || 'downloads',
    })
    
    const response = await fetch(`https://huggingface.co/api/models?${params}`)
    if (!response.ok) {
      throw new Error(`HuggingFace API error: ${response.statusText}`)
    }
    
    return response.json()
  }, 3600 * 1000) // 1 hour cache
}
```

**Same for CivitAI:**

```typescript
// bin/79_marketplace_core/marketplace-node/src/civitai.ts
export async function fetchCivitAIModels(options: any): Promise<CivitAIModel[]> {
  const cacheKey = `civitai-${JSON.stringify(options)}`
  
  return getCached(cacheKey, async () => {
    // ... existing fetch logic
  }, 3600 * 1000) // 1 hour cache
}
```

### Cache Invalidation

**Automatic:**
- TTL expires after 1 hour
- Fresh data fetched on next request

**Manual:**
```bash
# Clear marketplace cache
rm -rf bin/79_marketplace_core/marketplace-node/.cache

# Rebuild with fresh data
pnpm build
```

## 3. Security Hardening

### 3.1 CORS Updates

**Current Problem:**
- Only allows localhost origins
- Marketplace.rbee.dev can't fetch during SSG build

**Fix:**

```typescript
// src/index.ts
app.use(
  '/*',
  cors({
    origin: [
      // Development
      'http://localhost:7836',  // Hive UI
      'http://localhost:7822',  // Commercial
      'http://localhost:7823',  // Marketplace
      'http://127.0.0.1:7836',
      'http://127.0.0.1:7822',
      'http://127.0.0.1:7823',
      
      // Production
      'https://marketplace.rbee.dev',
      'https://rbee.dev',
      'https://docs.rbee.dev',
      
      // Cloudflare Pages preview URLs
      /https:\/\/.*\.rbee-marketplace\.pages\.dev$/,
      /https:\/\/.*\.rbee-commercial\.pages\.dev$/,
    ],
    allowMethods: ['GET', 'OPTIONS'], // Only read operations
    allowHeaders: ['Content-Type'],
    exposeHeaders: ['Content-Length', 'Cache-Control'],
    maxAge: 600,
    credentials: false, // No cookies needed
  }),
)
```

### 3.2 Security Headers

```typescript
// src/middleware/security.ts
import { Hono } from 'hono'

export function securityHeaders() {
  return async (c, next) => {
    await next()
    
    // Security headers
    c.header('X-Content-Type-Options', 'nosniff')
    c.header('X-Frame-Options', 'DENY')
    c.header('X-XSS-Protection', '1; mode=block')
    c.header('Referrer-Policy', 'strict-origin-when-cross-origin')
    c.header('Permissions-Policy', 'geolocation=(), microphone=(), camera=()')
    
    // CSP for API (strict)
    c.header(
      'Content-Security-Policy',
      "default-src 'none'; frame-ancestors 'none'"
    )
  }
}

// src/index.ts
import { securityHeaders } from './middleware/security'

app.use('/*', securityHeaders())
```

### 3.3 Input Validation

```typescript
// src/middleware/validation.ts
export function validateWorkerId() {
  return async (c, next) => {
    const id = c.req.param('id')
    
    // Validate worker ID format
    if (!/^[a-z0-9-]+$/.test(id)) {
      return c.json({ error: 'Invalid worker ID format' }, 400)
    }
    
    // Prevent path traversal
    if (id.includes('..') || id.includes('/')) {
      return c.json({ error: 'Invalid worker ID' }, 400)
    }
    
    await next()
  }
}

// src/routes.ts
routes.get('/workers/:id', validateWorkerId(), (c) => {
  // ... handler
})
```

### 3.4 Error Handling

```typescript
// src/middleware/errorHandler.ts
export function errorHandler() {
  return async (c, next) => {
    try {
      await next()
    } catch (error) {
      console.error('[error]', error)
      
      // Don't leak internal errors
      return c.json(
        {
          error: 'Internal server error',
          message: 'An unexpected error occurred',
        },
        500
      )
    }
  }
}

// src/index.ts
app.use('/*', errorHandler())
```

## 4. Monitoring & Observability

### 4.1 Request Logging

```typescript
// src/middleware/logging.ts
export function requestLogger() {
  return async (c, next) => {
    const start = Date.now()
    const method = c.req.method
    const path = c.req.path
    const ip = c.req.header('CF-Connecting-IP') || 'unknown'
    
    await next()
    
    const duration = Date.now() - start
    const status = c.res.status
    
    console.log(`[${method}] ${path} ${status} ${duration}ms (${ip})`)
  }
}
```

### 4.2 Cloudflare Analytics

Already enabled in `wrangler.jsonc`:
```json
{
  "observability": {
    "enabled": true
  }
}
```

**View in Dashboard:**
- Workers & Pages → gwc-rbee → Analytics
- See: Requests, Errors, Duration, Bandwidth

## 5. Implementation Plan

### Phase 1: Rate Limiting (High Priority)
1. Add Cloudflare Dashboard rate limiting rules
2. Test with load testing tool
3. Monitor in Cloudflare Analytics

### Phase 2: Caching (High Priority)
1. Add cache headers to worker catalog responses
2. Implement filesystem cache in marketplace-node
3. Test SSG build with cache
4. Verify cache invalidation works

### Phase 3: Security Hardening (Medium Priority)
1. Update CORS to allow production domains
2. Add security headers middleware
3. Add input validation
4. Add error handler

### Phase 4: Monitoring (Low Priority)
1. Add request logging
2. Set up Cloudflare alerts
3. Monitor error rates

## 6. Testing

### Rate Limiting Test
```bash
# Test rate limit (should block after 60 requests)
for i in {1..100}; do
  curl https://gwc.rbee.dev/workers
  sleep 0.5
done
```

### Cache Test
```bash
# First request (cache miss)
time curl https://gwc.rbee.dev/workers

# Second request (cache hit - should be faster)
time curl https://gwc.rbee.dev/workers

# Check cache headers
curl -I https://gwc.rbee.dev/workers | grep -i cache
```

### CORS Test
```bash
# Test from marketplace domain
curl -H "Origin: https://marketplace.rbee.dev" \
     -H "Access-Control-Request-Method: GET" \
     -X OPTIONS \
     https://gwc.rbee.dev/workers
```

## 7. Deployment

```bash
# 1. Update code with security features
cd bin/80-hono-worker-catalog

# 2. Test locally
pnpm dev
pnpm test

# 3. Deploy
cargo xtask deploy --app gwc --bump patch

# 4. Configure Cloudflare Dashboard
# - Add rate limiting rules
# - Verify analytics enabled
# - Check cache settings
```

## Summary

### Critical (Do Now)
- ✅ Rate limiting via Cloudflare Dashboard
- ✅ Add cache headers to responses
- ✅ Implement marketplace-node caching
- ✅ Update CORS for production domains

### Important (Do Soon)
- ⏭️ Add security headers
- ⏭️ Add input validation
- ⏭️ Add request logging

### Nice to Have (Do Later)
- ⏭️ Cloudflare KV for distributed rate limiting
- ⏭️ Advanced caching with stale-while-revalidate
- ⏭️ Metrics dashboard

This will make gwc.rbee.dev production-ready and prevent rate limit issues during marketplace builds!
