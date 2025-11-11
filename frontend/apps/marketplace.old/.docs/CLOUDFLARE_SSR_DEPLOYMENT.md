# Cloudflare Pages SSR Deployment Guide

**Date:** 2025-11-11  
**Team:** TEAM-475  
**Status:** ✅ READY FOR DEPLOYMENT

## Overview

The rbee marketplace now uses **Server-Side Rendering (SSR)** on Cloudflare Pages. This guide explains how to deploy and configure the marketplace for production.

## Prerequisites

1. **Cloudflare Pages project**: `rbee-marketplace`
2. **@cloudflare/next-on-pages**: Already configured in `next.config.ts`
3. **Wrangler CLI**: `pnpm add -D wrangler` (already installed)

## Build Process

### Local Build

```bash
cd frontend/apps/marketplace
pnpm run build
```

**What happens:**
1. Next.js builds the app with SSR enabled
2. Generates `.vercel/output/static` directory
3. Creates Cloudflare Workers for SSR pages
4. No manifest generation (removed in TEAM-475)

### Build Output

```
.vercel/output/
├── static/           # Static assets (CSS, JS, images)
├── functions/        # Cloudflare Workers for SSR pages
└── config.json       # Cloudflare Pages configuration
```

## Deployment

### Option 1: Automatic Deployment (Recommended)

```bash
pnpm run deploy
```

**What this does:**
1. Runs `pnpm run build` (builds Next.js app)
2. Runs `npx wrangler pages deploy .vercel/output/static`
3. Uploads to Cloudflare Pages project `rbee-marketplace`
4. Deploys to production branch `main`

### Option 2: Manual Deployment

```bash
# Build first
pnpm run build

# Deploy to Cloudflare Pages
npx wrangler pages deploy .vercel/output/static \
  --project-name=rbee-marketplace \
  --branch=main
```

### Option 3: CI/CD Deployment

Add to your CI/CD pipeline:

```yaml
# .github/workflows/deploy-marketplace.yml
name: Deploy Marketplace
on:
  push:
    branches: [main]
    paths:
      - 'frontend/apps/marketplace/**'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v2
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'pnpm'
      
      - name: Install dependencies
        run: pnpm install
      
      - name: Build marketplace
        run: |
          cd frontend/apps/marketplace
          pnpm run build
      
      - name: Deploy to Cloudflare Pages
        uses: cloudflare/wrangler-action@v3
        with:
          apiToken: ${{ secrets.CLOUDFLARE_API_TOKEN }}
          accountId: ${{ secrets.CLOUDFLARE_ACCOUNT_ID }}
          command: pages deploy .vercel/output/static --project-name=rbee-marketplace --branch=main
```

## Cloudflare Pages Configuration

### Environment Variables

Set these in Cloudflare Pages dashboard:

```bash
# Production
NEXT_PUBLIC_SITE_URL=https://marketplace.rbee.dev
NODE_ENV=production

# Optional: API rate limiting
HUGGINGFACE_API_KEY=<your-key>  # For higher rate limits
CIVITAI_API_KEY=<your-key>      # For higher rate limits
```

### Custom Domain

1. Go to Cloudflare Pages dashboard
2. Select `rbee-marketplace` project
3. Go to **Custom domains**
4. Add `marketplace.rbee.dev`
5. Cloudflare will automatically configure DNS

### Build Settings

**In Cloudflare Pages dashboard:**

- **Framework preset**: Next.js
- **Build command**: `pnpm run build`
- **Build output directory**: `.vercel/output/static`
- **Root directory**: `frontend/apps/marketplace`
- **Node version**: `20`

## SSR Performance

### Cloudflare Workers Limits

- **CPU time**: 50ms (free), 200ms (paid)
- **Memory**: 128MB
- **Request timeout**: 30 seconds

### Optimization Strategies

1. **Caching**: Add Cloudflare KV cache for popular models
2. **Edge caching**: Cloudflare CDN caches SSR responses automatically
3. **Parallel fetching**: Fetch model + compatibility data in parallel
4. **Error handling**: Graceful fallbacks for API failures

### Expected Performance

- **Page load**: 200-500ms (API call + render)
- **Cache hit**: 50-100ms (Cloudflare edge cache)
- **Build time**: 30 seconds (no manifest generation)

## Monitoring

### Cloudflare Analytics

1. Go to Cloudflare Pages dashboard
2. Select `rbee-marketplace` project
3. View **Analytics** tab

**Metrics to monitor:**
- Requests per second
- Response time (p50, p95, p99)
- Error rate (4xx, 5xx)
- Cache hit ratio

### Logs

View real-time logs:

```bash
npx wrangler pages deployment tail --project-name=rbee-marketplace
```

### Alerts

Set up alerts in Cloudflare dashboard:
- High error rate (>5%)
- Slow response time (>1s)
- High CPU usage (>80%)

## Troubleshooting

### Build Failures

**Error: "Module not found"**
```bash
# Clear cache and rebuild
rm -rf .next node_modules
pnpm install
pnpm run build
```

**Error: "WASM module not found"**
```bash
# Check WASM configuration in next.config.ts
# Ensure marketplace-node is installed
pnpm add @rbee/marketplace-node
```

### Deployment Failures

**Error: "Wrangler not found"**
```bash
# Install wrangler
pnpm add -D wrangler
```

**Error: "Unauthorized"**
```bash
# Login to Cloudflare
npx wrangler login
```

### Runtime Errors

**Error: "Worker exceeded CPU limit"**
- API calls are taking too long
- Add caching or reduce API calls
- Consider using Cloudflare KV for popular models

**Error: "Module not found at runtime"**
- Check that all dependencies are in `dependencies` (not `devDependencies`)
- Rebuild and redeploy

## Rollback

If SSR deployment fails, rollback to previous version:

```bash
# List deployments
npx wrangler pages deployment list --project-name=rbee-marketplace

# Rollback to specific deployment
npx wrangler pages deployment rollback <deployment-id> --project-name=rbee-marketplace
```

## Next Steps

1. **Deploy to staging first**: Test SSR in staging environment
2. **Monitor performance**: Track response times and error rates
3. **Add caching**: Implement Cloudflare KV cache for popular models
4. **Optimize API calls**: Batch requests, add retry logic
5. **Add error handling**: Graceful fallbacks for API failures

## References

- **Cloudflare Pages**: https://developers.cloudflare.com/pages/
- **Next.js on Cloudflare**: https://developers.cloudflare.com/pages/framework-guides/nextjs/ssr/
- **@cloudflare/next-on-pages**: https://github.com/cloudflare/next-on-pages
- **Wrangler CLI**: https://developers.cloudflare.com/workers/wrangler/

---

**Deployment guide created by TEAM-475 on 2025-11-11**
