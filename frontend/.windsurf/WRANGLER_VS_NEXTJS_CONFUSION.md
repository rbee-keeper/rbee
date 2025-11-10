# Wrangler vs Next.js Environment Configuration - THE CONFUSION

**Date:** 2025-11-10  
**Status:** 🔴 CRITICAL MISUNDERSTANDING

---

## 🎯 YOU'RE ABSOLUTELY RIGHT

**The apps ARE deployed with Wrangler/Cloudflare, but they're using Next.js SSG (Static Site Generation), NOT Cloudflare Workers!**

This is the source of massive confusion.

---

## 📊 THE ACTUAL ARCHITECTURE

### **What's Actually Happening:**

```
Commercial/Marketplace/User-Docs Apps:
├── Framework: Next.js 16
├── Build Mode: Static Export (output: 'export')
├── Deployment: Cloudflare Pages (NOT Workers)
├── Dev Server: Next.js dev server (NOT wrangler dev)
└── Environment: .env.local (Next.js standard, NOT .dev.vars)
```

### **Admin App (Different!):**

```
Admin App:
├── Framework: Hono + Cloudflare Workers
├── Build Mode: Workers runtime
├── Deployment: Cloudflare Workers
├── Dev Server: wrangler dev
└── Environment: .dev.vars (Cloudflare Workers standard)
```

---

## 🔍 EVIDENCE

### **1. Next.js Config Shows Static Export**

```typescript
// apps/commercial/next.config.ts
const nextConfig: NextConfig = {
  output: 'export',  // ⚠️ STATIC SITE GENERATION, NOT WORKERS
}

// apps/marketplace/next.config.ts
const nextConfig: NextConfig = {
  output: 'export',  // ⚠️ STATIC SITE GENERATION, NOT WORKERS
}
```

**What this means:**
- Next.js builds a **static HTML/CSS/JS site**
- No server-side rendering
- No Cloudflare Workers runtime
- Deployed to **Cloudflare Pages** (static hosting)
- Uses **Next.js environment variables** (`.env.local`, `NEXT_PUBLIC_*`)

### **2. Package.json Shows Next.js Dev Server**

```json
// apps/commercial/package.json
"dev:next": "next dev -p 7822"  // ⚠️ Next.js dev server, NOT wrangler

// apps/marketplace/package.json
"dev": "next dev --turbopack -p 7823"  // ⚠️ Next.js dev server, NOT wrangler

// apps/user-docs/package.json
"dev": "next dev -p 7811"  // ⚠️ Next.js dev server, NOT wrangler
```

**What this means:**
- Development uses `next dev` (Next.js dev server)
- NOT `wrangler dev` (Cloudflare Workers dev server)
- Environment variables come from `.env.local` (Next.js standard)
- NOT `.dev.vars` (Cloudflare Workers standard)

### **3. Deployment Uses Wrangler Pages (Static Hosting)**

```json
// apps/commercial/package.json
"deploy": "pnpm build && wrangler pages deploy out --project-name=rbee-commercial"

// apps/marketplace/package.json
"deploy": "pnpm run build && npx wrangler pages deploy out/ --project-name=rbee-marketplace"
```

**What this means:**
- `wrangler pages deploy` = Deploy static files to Cloudflare Pages
- NOT `wrangler deploy` (which would be for Workers)
- The `out/` folder contains **static HTML/CSS/JS** (from `next build`)
- Wrangler is just the **deployment tool**, not the runtime

### **4. .gitignore Shows Next.js Pattern**

```gitignore
# apps/commercial/.gitignore
.env*.local  # ⚠️ Next.js pattern

# wrangler files
.wrangler
.dev.vars*
!.dev.vars.example
```

**What this means:**
- `.env*.local` is gitignored (Next.js standard)
- `.dev.vars*` is ALSO gitignored (but not used for Next.js apps!)
- The `.dev.vars` pattern is there because someone copy-pasted from admin app

---

## 🤯 THE CONFUSION

### **Why `.dev.vars` Appears in .gitignore:**

Someone (probably copying from the admin app) added `.dev.vars*` to the commercial/marketplace/user-docs `.gitignore` files, even though **these apps don't use `.dev.vars`!**

```gitignore
# This is WRONG for Next.js SSG apps:
.dev.vars*
!.dev.vars.example
```

**These apps use Next.js, so they should use:**
```gitignore
.env*.local
!.env.example
```

### **Why There's No `.dev.vars` Files:**

**Because Next.js apps don't use `.dev.vars`!** They use `.env.local`.

The `.gitignore` pattern is misleading - it suggests `.dev.vars` should exist, but it shouldn't for Next.js apps.

---

## 📋 CORRECT ENVIRONMENT FILE USAGE

| App | Runtime | Dev Server | Env File | Deployment |
|-----|---------|------------|----------|------------|
| **commercial** | Next.js SSG | `next dev` | `.env.local` | Cloudflare Pages |
| **marketplace** | Next.js SSG | `next dev` | `.env.local` | Cloudflare Pages |
| **user-docs** | Next.js SSG | `next dev` | `.env.local` | Cloudflare Pages |
| **admin** | Cloudflare Workers | `wrangler dev` | `.dev.vars` | Cloudflare Workers |
| **global-worker-catalog** | Cloudflare Workers | `wrangler dev` | `.dev.vars` | Cloudflare Workers |

---

## 🎓 NEXT.JS ENVIRONMENT VARIABLES

### **How Next.js Handles Env Vars:**

1. **`.env.local`** - Local development (gitignored)
2. **`.env.development`** - Development defaults (committed)
3. **`.env.production`** - Production defaults (committed)
4. **`.env`** - All environments (committed)

### **Prefix Rules:**

- `NEXT_PUBLIC_*` - Exposed to browser (inlined at build time)
- No prefix - Server-side only (but SSG has no server!)

### **For SSG (Static Export):**

- **ALL env vars must use `NEXT_PUBLIC_*` prefix**
- They're inlined at build time into the static HTML/JS
- No runtime environment variables (it's static files!)

---

## 🔧 THE ACTUAL PROBLEM

### **Current State (WRONG):**

```bash
# apps/commercial/.env.local
NEXT_PUBLIC_MARKETPLACE_URL=https://marketplace.rbee.dev  # ❌ PRODUCTION
NEXT_PUBLIC_SITE_URL=https://rbee.dev                     # ❌ PRODUCTION
NEXT_PUBLIC_DOCS_URL=https://docs.rbee.dev                # ❌ PRODUCTION
```

### **Correct State (SHOULD BE):**

```bash
# apps/commercial/.env.local
NEXT_PUBLIC_MARKETPLACE_URL=http://localhost:7823  # ✅ LOCALHOST
NEXT_PUBLIC_SITE_URL=http://localhost:7822         # ✅ LOCALHOST
NEXT_PUBLIC_DOCS_URL=http://localhost:7811         # ✅ LOCALHOST
```

---

## 🎯 WHY THIS MATTERS

### **The Build Process:**

```bash
# Development (next dev)
1. Reads .env.local
2. Inlines NEXT_PUBLIC_* vars into code
3. Starts dev server on port 7822
4. @rbee/env-config reads process.env.NEXT_PUBLIC_SITE_URL
5. Returns "https://rbee.dev" (WRONG!)

# Production (next build)
1. Reads .env.production (or uses defaults)
2. Inlines NEXT_PUBLIC_* vars into static files
3. Generates static HTML/CSS/JS in out/
4. wrangler pages deploy uploads to Cloudflare Pages
```

### **The Problem:**

- `.env.local` has production URLs
- `@rbee/env-config` reads these and returns production URLs
- Even in development mode, links go to production
- Engineers hardcoded URLs as a workaround

---

## 🔍 WHAT HAPPENED TO `.dev.vars`?

**Nothing happened to them - they never existed for these apps!**

The `.gitignore` pattern `.dev.vars*` is a **red herring**. It was probably copy-pasted from the admin app's `.gitignore`.

### **Timeline (Hypothesis):**

1. Admin app created with Cloudflare Workers → Uses `.dev.vars`
2. Commercial/marketplace/user-docs apps created with Next.js SSG
3. Someone copy-pasted `.gitignore` from admin app
4. `.dev.vars*` pattern included, even though these apps use `.env.local`
5. Confusion ensues

---

## ✅ THE FIX

### **1. Clean Up .gitignore (Optional)**

Remove misleading `.dev.vars` patterns from Next.js apps:

```diff
# apps/commercial/.gitignore
- # wrangler files
- .wrangler
- .dev.vars*
- !.dev.vars.example
- !.env.example
```

### **2. Fix .env.local Files (REQUIRED)**

Update with localhost URLs:

```bash
# apps/commercial/.env.local
NEXT_PUBLIC_MARKETPLACE_URL=http://localhost:7823
NEXT_PUBLIC_SITE_URL=http://localhost:7822
NEXT_PUBLIC_DOCS_URL=http://localhost:7811
NEXT_PUBLIC_GITHUB_URL=https://github.com/rbee-keeper/rbee
NEXT_PUBLIC_LEGAL_EMAIL=legal@rbee.dev
NEXT_PUBLIC_SUPPORT_EMAIL=support@rbee.dev
```

### **3. Update .env.example Files (DOCUMENTATION)**

Make it clear what values should be used:

```bash
# apps/commercial/.env.example
# Development (use localhost)
NEXT_PUBLIC_MARKETPLACE_URL=http://localhost:7823
NEXT_PUBLIC_SITE_URL=http://localhost:7822
NEXT_PUBLIC_DOCS_URL=http://localhost:7811

# Production (set in Cloudflare Pages dashboard)
# NEXT_PUBLIC_MARKETPLACE_URL=https://marketplace.rbee.dev
# NEXT_PUBLIC_SITE_URL=https://rbee.dev
# NEXT_PUBLIC_DOCS_URL=https://docs.rbee.dev
```

---

## 📝 SUMMARY

| Question | Answer |
|----------|--------|
| Do these apps use Wrangler? | Yes, for **deployment only** (Cloudflare Pages) |
| Do these apps use `.dev.vars`? | **NO** - They use `.env.local` (Next.js standard) |
| Why is `.dev.vars` in `.gitignore`? | Copy-paste error from admin app |
| What happened to `.dev.vars`? | **Nothing - they never existed for Next.js apps** |
| What's the correct env file? | `.env.local` with `NEXT_PUBLIC_*` variables |
| Why are URLs wrong? | `.env.local` has production URLs instead of localhost |

---

**You're not stupid. The setup is confusing because:**
1. Wrangler is used for deployment (Cloudflare Pages)
2. But the apps run Next.js (not Cloudflare Workers)
3. `.gitignore` has misleading `.dev.vars` patterns
4. `.env.local` has wrong values (production instead of localhost)

**The fix is simple: Update `.env.local` files with localhost URLs.**
