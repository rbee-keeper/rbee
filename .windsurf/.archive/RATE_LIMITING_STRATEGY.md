# Rate Limiting Strategy for Manifest Generation

**Created**: 2025-11-11  
**Context**: TEAM-464 Masterplan Phase 2

---

## Why Rate Limiting?

When generating 25+ manifests (16 CivitAI + 9 HuggingFace), hitting APIs in parallel without limits can cause:

1. **Rate limit errors** - APIs reject requests (429 Too Many Requests)
2. **IP bans** - Temporary or permanent blocks
3. **Poor performance** - Server overload reduces response times
4. **Wasted resources** - Failed requests need retries

---

## Implementation Choice: TypeScript

**Decision**: Implement rate limiting in **TypeScript** (manifest generation script), not in WASM SDK.

### Why TypeScript?

✅ **Simple to implement** - No Rust async complexity  
✅ **Build-time only** - Only runs during manifest generation  
✅ **Easy to adjust** - Change concurrency/delay without recompiling WASM  
✅ **Visible progress** - Console logs show parallel execution  

### Why NOT in WASM?

❌ **WASM only used by Node.js** - No benefit to embedding in SDK  
❌ **Rust async complexity** - Tokio semaphores, rate limiters  
❌ **Harder to debug** - WASM black box  
❌ **Recompilation needed** - Every config change requires rebuild  

---

## RateLimiter Implementation

```typescript
class RateLimiter {
  private running = 0
  private maxConcurrent: number      // Max parallel requests
  private minDelay: number            // Min ms between requests
  private lastRun = 0

  constructor(maxConcurrent: number = 3, minDelayMs: number = 100) {
    this.maxConcurrent = maxConcurrent
    this.minDelay = minDelayMs
  }

  async run<T>(fn: () => Promise<T>): Promise<T> {
    // Wait if max concurrent requests running
    while (this.running >= this.maxConcurrent) {
      await new Promise(resolve => setTimeout(resolve, 50))
    }

    // Ensure minimum delay between requests
    const now = Date.now()
    const timeSinceLastRun = now - this.lastRun
    if (timeSinceLastRun < this.minDelay) {
      await new Promise(resolve => 
        setTimeout(resolve, this.minDelay - timeSinceLastRun)
      )
    }

    this.running++
    this.lastRun = Date.now()

    try {
      return await fn()
    } finally {
      this.running--
    }
  }
}
```

---

## Usage Pattern

```typescript
// Create limiter: max 3 concurrent, 100ms between each
const limiter = new RateLimiter(3, 100)

// Map all filters to promises
const promises = filters.map(filter => 
  limiter.run(async () => {
    // This runs with rate limiting
    return await fetchModelsViaSDK(filter)
  })
)

// Execute all in parallel (but rate-limited)
const results = await Promise.all(promises)
```

---

## Rate Limit Parameters

### Current Settings

- **Max Concurrent**: 3 requests at once
- **Min Delay**: 100ms between requests
- **Total time**: ~8-10 seconds for 25 filters

### Why These Values?

**Max Concurrent = 3**:
- HuggingFace: No official limit, but conservative
- CivitAI: Unknown limits, play it safe
- 3 is fast enough, safe enough

**Min Delay = 100ms**:
- 10 requests per second max
- Well below most API limits
- Barely noticeable to user

### Adjusting for Production

If rate limited, adjust:

```typescript
// More conservative (slower, safer)
const limiter = new RateLimiter(2, 200)  // 2 concurrent, 200ms delay

// More aggressive (faster, riskier)
const limiter = new RateLimiter(5, 50)   // 5 concurrent, 50ms delay
```

---

## Comparison: Sequential vs Parallel

### Sequential (Old Approach)
```typescript
for (const filter of filters) {
  await fetchModels(filter)
}
// Time: 25 filters × 1s each = 25 seconds
```

### Parallel Unlimited (Bad)
```typescript
const promises = filters.map(filter => fetchModels(filter))
await Promise.all(promises)
// Time: ~1 second (but may hit rate limits!)
```

### Parallel with Rate Limiting (✅ Best)
```typescript
const limiter = new RateLimiter(3, 100)
const promises = filters.map(filter => limiter.run(() => fetchModels(filter)))
await Promise.all(promises)
// Time: ~8 seconds (fast AND safe)
```

---

## Performance Expectations

| Filters | Sequential | Parallel (unlimited) | Parallel (rate-limited) |
|---------|-----------|----------------------|-------------------------|
| 10 | 10s | 1s | 3.5s |
| 25 | 25s | 1s | 8.5s |
| 50 | 50s | 1s | 17s |

**Sweet spot**: 3 concurrent with 100ms delay gives 3x speedup vs sequential while staying safe.

---

## Error Handling

```typescript
const promises = filters.map(filter =>
  limiter.run(async () => {
    try {
      return await fetchModelsViaSDK(filter)
    } catch (error) {
      console.error(`Failed to fetch ${filter}:`, error)
      return []  // Return empty instead of throwing
    }
  })
)

// All promises resolve (no rejections), some may have empty results
const results = await Promise.all(promises)
```

**Benefits**:
- One failed filter doesn't stop others
- Partial manifests still generated
- Clear error logging

---

## Monitoring

Add progress logging:

```typescript
let completed = 0
const total = filters.length

const promises = filters.map(filter =>
  limiter.run(async () => {
    const result = await fetchModelsViaSDK(filter)
    completed++
    console.log(`Progress: ${completed}/${total} (${Math.round(completed/total*100)}%)`)
    return result
  })
)
```

---

## Future Improvements

### 1. Retry Logic
```typescript
async function fetchWithRetry(filter: string, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fetchModelsViaSDK(filter)
    } catch (error) {
      if (i === maxRetries - 1) throw error
      await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)))
    }
  }
}
```

### 2. Adaptive Rate Limiting
```typescript
class AdaptiveRateLimiter extends RateLimiter {
  adjustOnError(error: Error) {
    if (error.message.includes('429')) {
      this.maxConcurrent = Math.max(1, this.maxConcurrent - 1)
      this.minDelay = this.minDelay * 2
      console.log(`Rate limited! Reduced to ${this.maxConcurrent} concurrent`)
    }
  }
}
```

### 3. Progress Bar
```typescript
import cliProgress from 'cli-progress'

const bar = new cliProgress.SingleBar()
bar.start(filters.length, 0)

// In limiter.run():
bar.increment()
```

---

## Summary

**Current Implementation**:
- ✅ TypeScript-based rate limiting
- ✅ 3 concurrent requests
- ✅ 100ms minimum delay
- ✅ ~8 seconds for 25 filters
- ✅ Safe from rate limits
- ✅ 3x faster than sequential

**Why Not WASM**:
- WASM only used by Node.js build scripts
- TypeScript implementation is simpler
- Easier to adjust and debug
- No recompilation needed for tuning

**Result**: Fast, safe, maintainable manifest generation at build time.
