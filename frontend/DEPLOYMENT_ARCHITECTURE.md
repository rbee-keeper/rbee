# Frontend Deployment Architecture

## Overview

The rbee frontend consists of three applications with different deployment strategies:

## 1. Marketplace (SSG - Static Site Generation)

**URL:** https://marketplace.rbee.dev  
**Technology:** Next.js with `output: 'export'`  
**Deployment:** Cloudflare Pages (static hosting)

### Configuration
- **No OpenNext** - Pure static export
- All pages pre-rendered at build time
- No server-side rendering
- No API routes

### Build Output
```bash
pnpm build
# Outputs to: out/
```

### Deployment
```bash
wrangler pages deploy out/ --project-name=rbee-marketplace --branch=main
```

### Why SSG?
- Model catalog is pre-generated from HuggingFace/CivitAI
- No dynamic content per user
- Fast CDN delivery
- Lower costs

---

## 2. User Docs (SSG - Static Site Generation)

**URL:** https://docs.rbee.dev  
**Technology:** Next.js + Nextra with `output: 'export'`  
**Deployment:** Cloudflare Pages (static hosting)

### Configuration
- **No OpenNext** - Pure static export
- All documentation pages pre-rendered
- No server-side rendering

### Build Output
```bash
pnpm build
# Outputs to: out/
```

### Deployment
```bash
wrangler pages deploy out/ --project-name=rbee-user-docs --branch=main
```

### Why SSG?
- Documentation is static content
- No user-specific content
- Fast page loads
- Easy CDN caching

---

## 3. Commercial Site (SSR - Server-Side Rendering)

**URL:** https://rbee.dev  
**Technology:** Next.js with OpenNext for Cloudflare  
**Deployment:** Cloudflare Workers + Pages

### Configuration
- **Uses OpenNext** - Full SSR support
- Server-side rendering enabled
- API routes supported
- Dynamic rendering

### Build Output
```bash
pnpm build
# Next.js builds to: .next/
# OpenNext transforms to: .open-next/
```

### Deployment
```bash
pnpm run deploy
# Runs: opennextjs-cloudflare build && opennextjs-cloudflare deploy
```

### Why SSR?
- Marketing site needs dynamic features
- Contact forms with server-side processing
- Potential user authentication
- A/B testing capabilities
- Personalized content

---

## Quick Reference

| Site | Type | OpenNext? | Output Dir | Deploy Command |
|------|------|-----------|------------|----------------|
| Marketplace | SSG | ❌ No | `out/` | `wrangler pages deploy out/` |
| User Docs | SSG | ❌ No | `out/` | `wrangler pages deploy out/` |
| Commercial | SSR | ✅ Yes | `.open-next/` | `opennextjs-cloudflare deploy` |

---

## xtask Deployment

All three can be deployed via xtask:

```bash
# SSG Sites
cargo xtask deploy marketplace
cargo xtask deploy docs

# SSR Site
cargo xtask deploy commercial
```

---

## Key Files

### Marketplace (SSG)
- `next.config.ts` - Has `output: 'export'`
- `package.json` - No `@opennextjs/cloudflare` dependency
- `wrangler.jsonc` - Simple Pages config

### User Docs (SSG)
- `next.config.ts` - Has `output: 'export'`
- `package.json` - No `@opennextjs/cloudflare` dependency
- `wrangler.jsonc` - Simple Pages config

### Commercial (SSR)
- `next.config.ts` - No `output: 'export'`, has OpenNext init
- `package.json` - Has `@opennextjs/cloudflare` dependency
- `open-next.config.ts` - OpenNext configuration
- `wrangler.jsonc` - Worker configuration with assets binding

---

## TEAM-XXX Summary

**Changes Made:**
1. ✅ Removed `@opennextjs/cloudflare` from marketplace dependencies
2. ✅ Removed `open-next.config.ts` from marketplace
3. ✅ Updated marketplace deploy script to use static Pages deployment
4. ✅ Fixed user-docs deployment script (was using wrong directory)
5. ✅ Updated commercial deployment to use `opennextjs-cloudflare deploy`

**Why This Matters:**
- SSG sites don't need OpenNext overhead
- Faster builds for static sites
- Lower costs (Pages is cheaper than Workers)
- Clearer separation of concerns
- Commercial site gets full SSR capabilities when needed
