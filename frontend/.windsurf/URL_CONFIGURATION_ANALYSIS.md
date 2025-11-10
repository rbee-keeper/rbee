# URL Configuration Analysis - CRITICAL ISSUE

**Date:** 2025-11-10  
**Status:** 🔴 CRITICAL - Production URLs hardcoded throughout codebase  
**Impact:** Development environment links to production URLs instead of localhost

---

## 🚨 THE PROBLEM

**During development, many links point to production URLs (`https://rbee.dev`, `https://marketplace.rbee.dev`, `https://docs.rbee.dev`) instead of localhost URLs.**

This breaks the development workflow and makes it impossible to test cross-app navigation locally.

---

## ✅ WHAT EXISTS (Good Foundation)

### 1. **@rbee/shared-config** - Port Configuration
**Location:** `/frontend/packages/shared-config/src/ports.ts`

**Purpose:** Single source of truth for port numbers
- ✅ Defines all ports (commercial: 7822, marketplace: 7823, docs: 7811, etc.)
- ✅ Environment-aware port selection
- ✅ Helper functions: `getIframeUrl()`, `getServiceUrl()`, `getWorkerUrl()`

**Limitation:** Only handles ports, not full URLs with domains

### 2. **@rbee/env-config** - Environment-Aware URLs
**Location:** `/frontend/packages/env-config/src/index.ts`

**Purpose:** Environment detection and URL generation
- ✅ Auto-detects `NODE_ENV` (development/production)
- ✅ Generates dev URLs: `http://localhost:7822`, `http://localhost:7823`, etc.
- ✅ Generates prod URLs: `https://rbee.dev`, `https://marketplace.rbee.dev`, etc.
- ✅ Exports `urls` object with helper functions

**Example:**
```typescript
import { urls, env } from '@rbee/env-config'

// Auto-switches based on NODE_ENV
urls.commercial        // Dev: http://localhost:7822, Prod: https://rbee.dev
urls.marketplace.home  // Dev: http://localhost:7823, Prod: https://marketplace.rbee.dev
urls.docs              // Dev: http://localhost:7811, Prod: https://docs.rbee.dev
```

---

## ❌ THE PROBLEM - Hardcoded Production URLs

### **Category 1: Navigation Configs** (CRITICAL)

#### **Commercial App Navigation**
**File:** `/frontend/apps/commercial/config/navigationConfig.ts`

**Lines 112-131 - Marketplace Links:**
```typescript
{
  label: 'HF Models',
  href: 'https://marketplace.rbee.dev/models/huggingface',  // ❌ HARDCODED
},
{
  label: 'CivitAi Models',
  href: 'https://marketplace.rbee.dev/models/civitai',      // ❌ HARDCODED
},
{
  label: 'Workers',
  href: 'https://marketplace.rbee.dev/workers',             // ❌ HARDCODED
},
cta: {
  label: 'Browse All',
  href: 'https://marketplace.rbee.dev',                     // ❌ HARDCODED
},
```

**Lines 160-161 - Docs Link:**
```typescript
docs: {
  url: 'https://docs.rbee.dev',  // ❌ HARDCODED
},
```

**Impact:** During development, clicking "Marketplace" or "Docs" in commercial app navigation takes you to production instead of localhost.

---

#### **User Docs Navigation**
**File:** `/frontend/apps/user-docs/config/navigationConfig.ts`

**Lines 5-11:**
```typescript
logoHref: 'https://rbee.dev',  // ❌ HARDCODED
links: [
  { label: 'Home', href: 'https://rbee.dev' },                    // ❌ HARDCODED
  { label: 'Marketplace', href: 'https://marketplace.rbee.dev' }, // ❌ HARDCODED
],
```

**Impact:** Clicking logo or "Home" in docs takes you to production commercial site instead of localhost.

---

### **Category 2: Footer Component** (CRITICAL)

**File:** `/frontend/packages/rbee-ui/src/organisms/Footer/Footer.tsx`

**Lines 44, 48, 57, 97, 102, 107, 113:**
```typescript
<a href="https://docs.rbee.dev">Documentation</a>                    // ❌ Line 44
<a href="https://github.com/rbee-keeper/rbee">Star on GitHub</a>    // ✅ OK (external)
<a href="https://discord.gg/rbee">Join Discord</a>                  // ✅ OK (external)

// Footer columns
{ href: 'https://docs.rbee.dev', text: 'Documentation' },           // ❌ Line 97
{ href: 'https://docs.rbee.dev/getting-started', text: 'Getting Started' }, // ❌ Line 102
{ href: 'https://marketplace.rbee.dev', text: 'Model Marketplace' }, // ❌ Line 107
{ href: 'https://github.com/rbee-keeper/rbee', text: 'GitHub' },    // ✅ OK (external)
```

**Impact:** Footer is shared across ALL apps (commercial, marketplace, user-docs). Every app's footer links to production URLs during development.

---

### **Category 3: CTA Components**

#### **InstallCTA Component**
**File:** `/frontend/apps/marketplace/components/InstallCTA.tsx`

**Lines 57, 66:**
```typescript
<a href="https://docs.rbee.dev/docs/getting-started/installation">
  Download rbee
</a>
<a href="https://rbee.dev">Learn More</a>  // ❌ HARDCODED
```

**Impact:** "Learn More" button in marketplace takes you to production instead of localhost.

---

#### **PopularModelsTemplate**
**File:** `/frontend/apps/commercial/components/templates/PopularModelsTemplate/PopularModelsTemplate.tsx`

**Lines 47, 50:**
```typescript
// Example in JSDoc
href: 'https://marketplace.rbee.dev/models/llama-3-8b'  // ❌ HARDCODED
viewAllHref="https://marketplace.rbee.dev/models"       // ❌ HARDCODED
```

**Impact:** Model cards link to production marketplace instead of localhost.

---

### **Category 4: Metadata (SEO)** (ACCEPTABLE)

**Files:**
- `/frontend/apps/commercial/app/layout.tsx` (line 36, 45)
- `/frontend/apps/marketplace/app/layout.tsx` (line 15, 30)
- `/frontend/apps/user-docs/app/layout.tsx` (line 11, 36)
- All page-level metadata (security, enterprise, features, etc.)

**Example:**
```typescript
metadataBase: new URL('https://rbee.dev'),  // ❌ HARDCODED (but acceptable for SEO)
openGraph: {
  url: 'https://rbee.dev/security',         // ❌ HARDCODED (but acceptable for SEO)
}
```

**Note:** These are acceptable because:
1. SEO metadata should always point to canonical production URLs
2. OpenGraph/Twitter cards need absolute URLs for social media crawlers
3. These don't affect user navigation

---

### **Category 5: Structured Data (SEO)** (ACCEPTABLE)

**File:** `/frontend/apps/commercial/lib/seo/structured-data.ts`

All Schema.org structured data contains production URLs. This is **acceptable** for the same reasons as metadata.

---

## 📊 SUMMARY OF ISSUES

| Category | Files Affected | Severity | Fix Priority |
|----------|---------------|----------|--------------|
| Navigation Configs | 2 files | 🔴 CRITICAL | 1 |
| Footer Component | 1 file (shared) | 🔴 CRITICAL | 1 |
| CTA Components | 2 files | 🟡 HIGH | 2 |
| Metadata (SEO) | ~20 files | 🟢 ACCEPTABLE | N/A |
| Structured Data | 1 file | 🟢 ACCEPTABLE | N/A |

---

## ✅ THE SOLUTION

### **Phase 1: Enhance @rbee/env-config** (DONE)

The package already exists and works correctly:

```typescript
import { urls, env } from '@rbee/env-config'

// Auto-switches based on NODE_ENV
urls.commercial        // Dev: http://localhost:7822, Prod: https://rbee.dev
urls.marketplace.home  // Dev: http://localhost:7823, Prod: https://marketplace.rbee.dev
urls.docs              // Dev: http://localhost:7811, Prod: https://docs.rbee.dev
urls.github.repo       // Always: https://github.com/veighnsche/llama-orch
```

### **Phase 2: Update Navigation Configs** (TODO)

**File:** `/frontend/apps/commercial/config/navigationConfig.ts`

**BEFORE:**
```typescript
{
  label: 'HF Models',
  href: 'https://marketplace.rbee.dev/models/huggingface',
}
```

**AFTER:**
```typescript
import { urls } from '@rbee/env-config'

{
  label: 'HF Models',
  href: `${urls.marketplace.home}/models/huggingface`,
}
```

**Apply to:**
- `/frontend/apps/commercial/config/navigationConfig.ts` (lines 112-131, 160-161)
- `/frontend/apps/user-docs/config/navigationConfig.ts` (lines 5-11)

---

### **Phase 3: Update Footer Component** (TODO)

**File:** `/frontend/packages/rbee-ui/src/organisms/Footer/Footer.tsx`

**BEFORE:**
```typescript
<a href="https://docs.rbee.dev">Documentation</a>
```

**AFTER:**
```typescript
import { urls } from '@rbee/env-config'

<a href={urls.docs}>Documentation</a>
```

**Apply to all hardcoded URLs in Footer (lines 44, 97, 102, 107)**

---

### **Phase 4: Update CTA Components** (TODO)

**Files:**
- `/frontend/apps/marketplace/components/InstallCTA.tsx`
- `/frontend/apps/commercial/components/templates/PopularModelsTemplate/PopularModelsTemplate.tsx`

**Same pattern:** Replace hardcoded URLs with `urls` from `@rbee/env-config`

---

## 🎯 IMPLEMENTATION CHECKLIST

- [ ] **Navigation Configs**
  - [ ] Update `/frontend/apps/commercial/config/navigationConfig.ts`
  - [ ] Update `/frontend/apps/user-docs/config/navigationConfig.ts`
  - [ ] Update `/frontend/apps/marketplace/config/navigationConfig.ts` (if needed)

- [ ] **Footer Component**
  - [ ] Update `/frontend/packages/rbee-ui/src/organisms/Footer/Footer.tsx`
  - [ ] Make Footer accept optional `urls` prop for environment awareness

- [ ] **CTA Components**
  - [ ] Update `/frontend/apps/marketplace/components/InstallCTA.tsx`
  - [ ] Update `/frontend/apps/commercial/components/templates/PopularModelsTemplate/PopularModelsTemplate.tsx`

- [ ] **Testing**
  - [ ] Start all apps in dev mode (`pnpm dev`)
  - [ ] Verify navigation links go to localhost
  - [ ] Verify footer links go to localhost
  - [ ] Verify CTA buttons go to localhost
  - [ ] Build for production and verify URLs switch to production

---

## 🔍 VERIFICATION COMMANDS

```bash
# Start all apps in dev mode
cd /home/vince/Projects/rbee/frontend
pnpm dev

# Check commercial app (should be localhost:7822)
# Click "Marketplace" in nav → should go to localhost:7823
# Click "Docs" in nav → should go to localhost:7811
# Click "Documentation" in footer → should go to localhost:7811

# Check marketplace app (should be localhost:7823)
# Click "Back to rbee.dev" → should go to localhost:7822
# Click "Docs" in nav → should go to localhost:7811

# Check user-docs app (should be localhost:7811)
# Click logo → should go to localhost:7822
# Click "Marketplace" → should go to localhost:7823
```

---

## 📝 NOTES

1. **@rbee/env-config already exists and works correctly** - No need to create it
2. **SEO metadata URLs are acceptable** - They should always point to production for crawlers
3. **External links (GitHub, Discord, Twitter) are fine** - They're always external
4. **The fix is straightforward** - Just replace hardcoded URLs with `urls` from `@rbee/env-config`

---

## 🎓 LESSONS LEARNED

1. **Centralized configuration exists but wasn't used** - The infrastructure was already in place
2. **Navigation configs were created without checking for existing solutions**
3. **Footer component was built with hardcoded URLs instead of environment-aware URLs**
4. **Need to enforce usage of @rbee/env-config in code reviews**

---

**Next Steps:** Implement Phase 2-4 to fix all hardcoded URLs in navigation, footer, and CTA components.
