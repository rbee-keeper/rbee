# Port Configuration

**Canonical Source of Truth for All Port Assignments**

**Last Updated:** 2025-11-09  
**Version:** 3.0

---

## Overview

This document is the **single source of truth** for all port assignments in the rbee ecosystem.

**When adding a new service:**
1. Update this file first
2. Update `frontend/packages/shared-config/src/ports.ts`
3. Run `pnpm generate:rust` (if applicable)
4. Update backend Cargo.toml with default port

---

## Port Assignments

### Desktop Applications

| Service | Dev Port | Production | Description |
|---------|----------|------------|-------------|
| **rbee-keeper** | 7843 | N/A (Tauri) | Desktop CLI manager |

**Notes:**
- Tauri desktop application
- No production HTTP port
- Dev port is Vite dev server

---


### Backend Services (HTTP APIs)

| Service | Dev Port | Prod Port | Production URL | Description |
|---------|----------|-----------|----------------|-------------|
| **queen-rbee** | 7844 (UI) | 7833 | {localhost}:7833 | Orchestrator API + UI |
| **rbee-hive** | 7845 (UI) | 7834 | {localhost/remote host}:7834 | Worker manager API + UI |

**Notes:**
- Dev ports are Vite dev servers
- Prod ports serve embedded UI from backend
- Workers are dynamically assigned ports by hive (starting from 7855)

---

### Frontend Applications (Cloudflare Deployments)

| Service | Dev Port | Production URL | Description |
|---------|----------|----------------|-------------|
| **commercial** | 7822 | https://rbee.dev | Commercial site (Next.js) |
| **marketplace** | 7823 | https://marketplace.rbee.dev | Marketplace (Next.js) |
| **user-docs** | 7824 | https://docs.rbee.dev | Documentation (Next.js) |
| **admin** | 7825 | https://backend.rbee.dev | Admin dashboard (Next.js) |

**Notes:**
- All deployed to Cloudflare Pages/Workers
- Dev ports are for local development only
- Production uses Cloudflare's infrastructure

---

### Cloudflare Workers

| Service | Dev Port | Production URL | Description |
|---------|----------|----------------|-------------|
| **global-worker-catalog** | 7811 | https://gwc.rbee.dev | Worker catalog API (Hono) |

**Notes:**
- Both use Cloudflare Workers runtime
- Dev ports are for `wrangler dev`
- Production uses Cloudflare Workers

---

### Storybooks

| Service | Dev Port | Description |
|---------|----------|-------------|
| **@rbee/ui** | 6006 | UI component library |
| **commercial** | 6007 | Commercial site components |

---

### Workers (Dynamic Ports)

| Worker Type | Dev Port | Prod Port | Description |
|-------------|----------|-----------|-------------|
| **llm-worker** | 7855 - 7899 | Dynamic | LLM inference worker |
| **sd-worker** | 7855 - 7899 | Dynamic | Stable Diffusion worker |

**Notes:**
- Production ports are **dynamically assigned** by hive (starting from 7855)
- Dev ports are fixed for local development
- Query hive for actual worker URLs in production

---

## Port Ranges

### Reserved Ranges

| Range | Purpose |
|-------|---------|
| 7843-7845 | Vite dev servers |
| 6006-6007 | Storybooks |
| 7822-7825 | Frontend apps (Nextjs) |
| 7833-7834 | Backend services + worker dev ports |
| 7855-7899 | Dynamic worker assignments (production) |
| 7811 | Hono Cloudflare Workers dev |

---

## Production URLs

### Public-Facing

| Service | URL | Type |
|---------|-----|------|
| **Commercial Site** | https://rbee.dev | Nextjs Cloudflare Pages |
| **Marketplace** | https://marketplace.rbee.dev | Nextjs Cloudflare Workers |
| **Documentation** | https://docs.rbee.dev | Nextjs Cloudflare Pages |
| **Admin Dashboard** | https://backend.rbee.dev | Nextjs Cloudflare Workers |
| **Worker Catalog** | https://gwc.rbee.dev | Hono Cloudflare Workers |

### Internal/Development

| Service | URL | Type |
|---------|-----|------|
| **Queen API** | http://localhost:7833 | Backend |
| **Hive API** | http://localhost:7834 | Backend |
| **Workers** | Dynamic (7855-7899) | Backend |

---

## Environment Variables

### Frontend Apps

```bash
# Commercial (7822)
VITE_COMMERCIAL_PORT=7822
NEXT_PUBLIC_SITE_URL=http://localhost:7822

# Marketplace (7823)
VITE_MARKETPLACE_PORT=7823
NEXT_PUBLIC_SITE_URL=http://localhost:7823

# User Docs (7824)
VITE_USER_DOCS_PORT=7824
NEXT_PUBLIC_SITE_URL=http://localhost:7824

# Admin Dashboard (7825)
VITE_ADMIN_PORT=7825
NEXT_PUBLIC_SITE_URL=http://localhost:7825
```

### Cloudflare Workers

```bash
# Global Worker Catalog (7811)
VITE_HONO_CATALOG_PORT=7811
```

### Backend Services

```bash
# Queen (7833 prod, 7844 dev UI)
VITE_QUEEN_PORT=7833
VITE_QUEEN_UI_DEV_PORT=7844

# Hive (7834 prod, 7845 dev UI)
VITE_HIVE_PORT=7834
VITE_HIVE_UI_DEV_PORT=7845
```

---

## Adding a New Service

### Checklist

1. ✅ **Choose an available port** from the reserved ranges
2. ✅ **Update this file** with the new service
3. ✅ **Update `frontend/packages/shared-config/src/ports.ts`**
4. ✅ **Run `pnpm build` in shared-config**
5. ✅ **Run `pnpm generate:rust`** (if backend integration needed)
6. ✅ **Update backend Cargo.toml** with default port
7. ✅ **Update `.env.example`** with environment variables
8. ✅ **Test locally** to verify port is available

---

## Port Conflicts

### How to Check

```bash
# Check if port is in use
lsof -i :PORT_NUMBER

# Kill process on port
kill -9 $(lsof -t -i:PORT_NUMBER)
```

### Common Conflicts

- **5173**: Vite default port (used by keeper)
- **3000**: Next.js default (NOT USED - we use 7822)
- **8080**: Common dev port (used by hive for workers)

---

## History

### v3.0 (2025-11-09)
- Added admin dashboard (8788, https://backend.rbee.dev)
- Documented production URLs for all services
- Added Cloudflare Workers section

### v2.0 (2025-10-27)
- Added commercial, marketplace, user-docs
- Added global-worker-catalog
- Restructured for clarity

### v1.0 (2025-10-25)
- Initial port configuration
- Backend services (queen, hive)
- Worker ports

---

**Remember:** This file is the canonical source of truth. Always update it first!
