# ROOT CAUSE ANALYSIS - URL Configuration Disaster

**Date:** 2025-11-10  
**Severity:** 🔴 CRITICAL  
**Status:** ROOT CAUSE IDENTIFIED

---

## 🎯 EXECUTIVE SUMMARY

**You were 100% correct.** The `.env.local` files contain **hardcoded production URLs** instead of localhost URLs. This broke the entire environment-aware URL system, forcing engineers to hardcode production URLs everywhere as a workaround.

---

## 🔍 THE SMOKING GUN

### **Actual .env.local Files (WRONG)**

```bash
# /home/vince/Projects/rbee/frontend/apps/commercial/.env.local
NEXT_PUBLIC_MARKETPLACE_URL=https://marketplace.rbee.dev  # ❌ PRODUCTION
NEXT_PUBLIC_SITE_URL=https://rbee.dev                     # ❌ PRODUCTION
NEXT_PUBLIC_DOCS_URL=https://docs.rbee.dev                # ❌ PRODUCTION
NEXT_PUBLIC_GITHUB_URL=https://github.com/rbee-keeper/rbee
NEXT_PUBLIC_LEGAL_EMAIL=legal@rbee.dev
NEXT_PUBLIC_SUPPORT_EMAIL=support@rbee.dev

# /home/vince/Projects/rbee/frontend/apps/marketplace/.env.local
MARKETPLACE_API_URL=https://gwc.rbee.dev                  # ❌ PRODUCTION
NEXT_PUBLIC_SITE_URL=https://rbee.dev                     # ❌ PRODUCTION
NEXT_DISABLE_DEVTOOLS=1

# /home/vince/Projects/rbee/frontend/apps/user-docs/.env.local
NEXT_PUBLIC_SITE_URL=https://docs.rbee.dev                # ❌ PRODUCTION
NEXT_PUBLIC_GITHUB_URL=https://github.com/rbee-keeper/rbee
```

### **What They SHOULD Contain (CORRECT)**

```bash
# /home/vince/Projects/rbee/frontend/apps/commercial/.env.local
NEXT_PUBLIC_MARKETPLACE_URL=http://localhost:7823         # ✅ LOCALHOST
NEXT_PUBLIC_SITE_URL=http://localhost:7822                # ✅ LOCALHOST
NEXT_PUBLIC_DOCS_URL=http://localhost:7811                # ✅ LOCALHOST
NEXT_PUBLIC_GITHUB_URL=https://github.com/rbee-keeper/rbee  # ✅ External OK
NEXT_PUBLIC_LEGAL_EMAIL=legal@rbee.dev                    # ✅ Email OK
NEXT_PUBLIC_SUPPORT_EMAIL=support@rbee.dev                # ✅ Email OK

# /home/vince/Projects/rbee/frontend/apps/marketplace/.env.local
MARKETPLACE_API_URL=http://localhost:8787                 # ✅ LOCALHOST
NEXT_PUBLIC_SITE_URL=http://localhost:7823                # ✅ LOCALHOST
NEXT_DISABLE_DEVTOOLS=1                                   # ✅ OK

# /home/vince/Projects/rbee/frontend/apps/user-docs/.env.local
NEXT_PUBLIC_SITE_URL=http://localhost:7811                # ✅ LOCALHOST
NEXT_PUBLIC_GITHUB_URL=https://github.com/rbee-keeper/rbee  # ✅ External OK
```

---

## 💥 HOW THIS BROKE EVERYTHING

### **The Chain of Failure**

1. **Someone created `.env.local` files with production URLs** (probably copy-pasted from production config)
2. **`@rbee/env-config` reads env vars FIRST** (before checking `NODE_ENV`)
3. **Even in development mode, production URLs are returned**
4. **Engineers noticed links going to production**
5. **Engineers hardcoded production URLs everywhere** (as a "workaround")
6. **The problem spread across navigation, footer, CTAs, etc.**

### **The Broken Code Flow**

```typescript
// @rbee/env-config/src/index.ts (lines 56-69)
function getUrl(key: keyof typeof PROD_URLS, envVar?: string): string {
  // 1. ⚠️ CHECK ENV VAR FIRST (HIGHEST PRIORITY)
  if (envVar && process.env[envVar]) {
    return process.env[envVar]  
    // ❌ Returns "https://marketplace.rbee.dev" from .env.local
    // ❌ Even though we're in development mode!
  }

  // 2. ✅ AUTO-DETECT BASED ON NODE_ENV (NEVER REACHED)
  if (isDev) {
    return DEV_URLS[key]  
    // ✅ Would return "http://localhost:7823"
    // ✅ But this code never runs because step 1 already returned!
  }

  // 3. FALLBACK TO PRODUCTION
  return PROD_URLS[key]
}
```

**The logic is CORRECT, but the `.env.local` files have WRONG values!**

---

## 📊 VERIFICATION - PORT CONFIGURATION

### **Canonical Source: PORT_CONFIGURATION.md**

| Service | Dev Port | Production URL | Status |
|---------|----------|----------------|--------|
| commercial | 7822 | https://rbee.dev | ✅ Documented |
| marketplace | 7823 | https://marketplace.rbee.dev | ✅ Documented |
| user-docs | 7811 | https://docs.rbee.dev | ✅ Documented |
| global-worker-catalog | 8787 | https://gwc.rbee.dev | ✅ Documented |
| admin | 8788 | https://install.rbee.dev | ✅ Documented |

### **Shared Config: @rbee/shared-config**

```typescript
// frontend/packages/shared-config/src/ports.ts
export const PORTS = {
  commercial: {
    dev: 7822,  // ✅ CORRECT
    prod: null, // Deployed to Cloudflare
  },
  marketplace: {
    dev: 7823,  // ✅ CORRECT
    prod: null, // Deployed to Cloudflare
  },
  userDocs: {
    dev: 7811,  // ✅ CORRECT
    prod: null, // Deployed to Cloudflare
  },
  honoCatalog: {
    dev: 8787,  // ✅ CORRECT
    prod: null, // Deployed to Cloudflare
  },
}
```

**✅ Shared config is CORRECT!**

### **Environment Config: @rbee/env-config**

```typescript
// frontend/packages/env-config/src/index.ts
const PROD_URLS = {
  commercial: 'https://rbee.dev',                    // ✅ CORRECT
  marketplace: 'https://marketplace.rbee.dev',       // ✅ CORRECT
  docs: 'https://docs.rbee.dev',                     // ✅ CORRECT
  github: 'https://github.com/veighnsche/llama-orch', // ✅ CORRECT
}

const DEV_URLS = {
  commercial: `http://localhost:${PORTS.commercial}`,   // ✅ CORRECT (7822)
  marketplace: `http://localhost:${PORTS.marketplace}`, // ✅ CORRECT (7823)
  docs: `http://localhost:${PORTS.userDocs}`,          // ✅ CORRECT (7811)
  github: PROD_URLS.github,                            // ✅ CORRECT (always external)
}
```

**✅ Environment config is CORRECT!**

### **Package.json Scripts**

```json
// commercial/package.json
"dev:next": "next dev -p 7822"  // ✅ CORRECT PORT

// marketplace/package.json
"dev": "next dev --turbopack -p 7823"  // ✅ CORRECT PORT

// user-docs/package.json
"dev": "next dev -p 7811"  // ✅ CORRECT PORT
```

**✅ Dev scripts are CORRECT!**

---

## 🎓 WHY ENGINEERS HARDCODED URLS

**They weren't being lazy. They were working around a broken config.**

### **Timeline of Events (Hypothesis)**

1. **Initial Setup:** Someone created `.env.local` files for development
2. **Copy-Paste Error:** They copied production URLs instead of localhost URLs
3. **Testing:** Engineers ran `pnpm dev` and clicked links
4. **Bug Discovery:** Links went to production instead of localhost
5. **Investigation:** Engineers checked `@rbee/env-config` and saw it was "broken"
6. **Workaround:** Engineers hardcoded production URLs directly in components
7. **Spread:** The pattern spread across navigation, footer, CTAs, etc.

### **Evidence from Code Comments**

```typescript
// commercial/config/navigationConfig.ts (line 112)
{
  label: 'HF Models',
  href: 'https://marketplace.rbee.dev/models/huggingface',  // No comment explaining why
}

// Footer.tsx (line 44)
<a href="https://docs.rbee.dev">Documentation</a>  // No comment explaining why

// InstallCTA.tsx (line 66)
<a href="https://rbee.dev">Learn More</a>  // No comment explaining why
```

**No comments = They didn't realize it was wrong. They thought this was the correct way.**

---

## 🔧 THE FIX

### **Phase 1: Fix .env.local Files** (IMMEDIATE)

Replace production URLs with localhost URLs in all `.env.local` files.

### **Phase 2: Remove Hardcoded URLs** (AFTER PHASE 1)

Once `.env.local` is fixed, `@rbee/env-config` will work correctly. Then we can:
1. Update navigation configs to use `urls` from `@rbee/env-config`
2. Update Footer component to use `urls`
3. Update CTA components to use `urls`

### **Phase 3: Add Validation** (PREVENT FUTURE ISSUES)

Add a dev-time check that warns if `.env.local` contains production URLs:

```typescript
// @rbee/env-config/src/index.ts
if (isDev && typeof window === 'undefined') {
  const hasProductionUrls = 
    process.env.NEXT_PUBLIC_SITE_URL?.includes('rbee.dev') ||
    process.env.NEXT_PUBLIC_MARKETPLACE_URL?.includes('rbee.dev') ||
    process.env.NEXT_PUBLIC_DOCS_URL?.includes('rbee.dev')
  
  if (hasProductionUrls) {
    console.warn('⚠️  WARNING: .env.local contains production URLs!')
    console.warn('⚠️  Development links will go to production instead of localhost.')
    console.warn('⚠️  Update .env.local to use localhost URLs.')
  }
}
```

---

## 📋 SUMMARY

| Component | Status | Issue |
|-----------|--------|-------|
| PORT_CONFIGURATION.md | ✅ CORRECT | Canonical source is accurate |
| @rbee/shared-config | ✅ CORRECT | Port numbers are correct |
| @rbee/env-config | ✅ CORRECT | Logic is correct |
| package.json scripts | ✅ CORRECT | Dev ports are correct |
| .env.local files | ❌ **WRONG** | **Production URLs instead of localhost** |
| Navigation configs | ❌ WRONG | Hardcoded as workaround |
| Footer component | ❌ WRONG | Hardcoded as workaround |
| CTA components | ❌ WRONG | Hardcoded as workaround |

---

## 🎯 ACTION ITEMS

1. **Fix .env.local files** (3 files, 5 minutes)
2. **Test that @rbee/env-config works** (visit debug pages)
3. **Remove hardcoded URLs** (navigation, footer, CTAs)
4. **Add validation warning** (prevent future issues)
5. **Update .env.local.example files** (document correct values)

---

## 🏆 CONCLUSION

**You were right to be suspicious.** The shared config infrastructure is actually excellent - it's just that the `.env.local` files were configured incorrectly, breaking the entire system.

**The engineers weren't wrong to hardcode URLs - they were working around what appeared to be a broken config system.**

**The fix is simple: Update 3 .env.local files with localhost URLs.**

---

**Next Step:** Do you want me to fix the `.env.local` files now?
