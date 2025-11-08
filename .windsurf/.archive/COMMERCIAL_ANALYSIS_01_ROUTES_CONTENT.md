# Commercial Site: Routes & Content Map

**Analysis Date:** 2025-11-06  
**Scope:** `frontend/apps/commercial`  
**Purpose:** Complete route inventory with content mapping

---

## Route Structure Overview

**Total Routes Found:** 32 page.tsx files  
**Route Organization:**
- Top-level pages: 11
- `/compare/*` subroutes: 4
- `/features/*` subroutes: 6
- `/industries/*` subroutes: 10
- `/legal/*` subroutes: 3

---

## Top-Level Routes

### `/` (Home)
**File:** `app/page.tsx`  
**Component:** `HomePage` from `components/pages`  
**Purpose:** Primary landing page with full marketing funnel  
**Audience:** All (consumer + business)  
**Related PART doc:** PART_01_HomePage.md

**Content Sections (from HomePage.tsx):**
1. Hero (WhatIsRbee)
2. 6 Unique Advantages (CardGridTemplate)
3. Popular Models (marketplace linking)
4. Problem/Solution
5. Before/After comparison
6. How It Works
7. Features Tabs
8. Use Cases
9. Comparison (vs competitors)
10. Pricing
11. Email Capture
12. Technical Details
13. FAQ
14. CTA

**Testimonials/Claims:**
- "€499 vs $72K+/year" (pricing comparison)
- "99.3% cheaper than cloud"
- "87% GPU utilization vs 45%" (appears in metadata, likely in content)
- "Turn your GPUs into ONE unified colony"

**SEO Metadata:**
- Title: "rbee - Stop Paying for AI APIs. Run Everything Free."
- Description: Aggressive CTR-focused (uses 🐝 emoji)
- Keywords: 12 terms including "rbee vs Ollama", "rbee vs vLLM"
- JSON-LD: Organization + WebSite schemas

**Obvious Duplication With:**
- `/features` - 6 Advantages section duplicated
- `/use-cases` - Use cases section duplicated
- `/industries/homelab` - Homelab content duplicated
- `/pricing` - Pricing section duplicated

---

### `/pricing`
**File:** `app/pricing/page.tsx`  
**Component:** `PricingPage`  
**Purpose:** Pricing tiers and lifetime licensing details  
**Audience:** All (conversion-focused)  
**Related PART doc:** PART_02_PricingPage.md

**Testimonials/Claims:**
- "Free forever (GPL-3.0)"
- "€129-499 lifetime"
- "99.3% cheaper than cloud ($72K+/year)"
- "Pay once, own forever. No subscriptions."

**SEO Metadata:**
- Title: "rbee Pricing - Free Forever or €129-499 Lifetime | No Subscriptions"
- Description: Emphasizes lifetime pricing, no subscriptions
- Keywords: 11 terms including "€499 vs $72K", "Together.ai alternative"
- JSON-LD: Product schemas for Premium Queen, GDPR, Complete Bundle
- OpenGraph + Twitter cards configured

**Obvious Duplication With:**
- `/` (Home) - Pricing section on homepage

---

### `/features`
**File:** `app/features/page.tsx`  
**Component:** `FeaturesPage`  
**Purpose:** Overview of all 6 core features  
**Audience:** Technical decision-makers  
**Related PART doc:** PART_03_FeaturesPage.md

**Obvious Duplication With:**
- `/` (Home) - 6 Advantages section
- `/features/*` subroutes - Each feature has dedicated page

**Note:** Has 6 dedicated subroutes (see below)

---

### `/use-cases`
**File:** `app/use-cases/page.tsx`  
**Component:** `UseCasesPage`  
**Purpose:** Use case overview/hub  
**Audience:** All  
**Related PART doc:** PART_06_UseCasePages.md (partial)

**Obvious Duplication With:**
- `/` (Home) - Use cases section
- `/industries/*` pages - Same content organized differently

---

### `/community`
**File:** `app/community/page.tsx`  
**Component:** `CommunityPage`  
**Purpose:** Community engagement, Discord, GitHub  
**Audience:** Developers, contributors  
**Related PART doc:** None (not in audit)

---

### `/developers`
**File:** `app/developers/page.tsx`  
**Component:** `DevelopersPage`  
**Purpose:** Developer-focused landing  
**Audience:** Developers  
**Related PART doc:** None (not in audit)

**Note:** Also exists as `/industries/developers` - DUPLICATION

---

### `/enterprise`
**File:** `app/enterprise/page.tsx`  
**Component:** `EnterprisePage`  
**Purpose:** Enterprise sales landing  
**Audience:** Enterprise buyers  
**Related PART doc:** None (not in audit)

**Note:** Also exists as `/industries/enterprise` - DUPLICATION

---

### `/gpu-providers`
**File:** `app/gpu-providers/page.tsx`  
**Component:** `ProvidersPage`  
**Purpose:** Earn money by renting GPUs  
**Audience:** GPU owners  
**Related PART doc:** PART_06_UseCasePages.md

**Testimonials/Claims:**
- "€180K+ paid to providers" (UNSOURCED - pre-launch impossible)
- Generic provider testimonials (likely fabricated)

**Obvious Duplication With:**
- `/industries/providers` - Same content

---

### `/security`
**File:** `app/security/page.tsx`  
**Component:** `SecurityPage`  
**Purpose:** Security features and compliance  
**Audience:** Enterprise, compliance officers  
**Related PART doc:** None (not in audit)

---

### `/legal`
**File:** `app/legal/page.tsx`  
**Component:** `LegalPage`  
**Purpose:** Legal hub/overview  
**Audience:** All (compliance)  
**Related PART doc:** None (not in audit)

**Note:** Has 2 subroutes: `/legal/privacy`, `/legal/terms`

---

## `/compare/*` Routes (4 pages)

### `/compare/rbee-vs-ollama`
**File:** `app/compare/rbee-vs-ollama/page.tsx`  
**Component:** `RbeeVsOllamaPage`  
**Purpose:** Competitive comparison vs Ollama  
**Audience:** Users evaluating Ollama  
**Related PART doc:** PART_04_ComparisonPages.md

**SEO Value:** HIGH (9/10 in audit)  
**AI-Tone Severity:** LOW (3/10)  
**Authority:** 6/10

**Note:** Audit calls this "excellent competitor comparison"

---

### `/compare/rbee-vs-vllm`
**File:** `app/compare/rbee-vs-vllm/page.tsx`  
**Component:** `RbeeVsVllmPage`  
**Purpose:** Competitive comparison vs vLLM  
**Audience:** Users evaluating vLLM  
**Related PART doc:** PART_04_ComparisonPages.md

---

### `/compare/rbee-vs-together-ai`
**File:** `app/compare/rbee-vs-together-ai/page.tsx`  
**Component:** `RbeeVsTogetherAiPage`  
**Purpose:** Competitive comparison vs Together.ai  
**Audience:** Users evaluating Together.ai  
**Related PART doc:** PART_04_ComparisonPages.md

---

### `/compare/rbee-vs-ray-kserve`
**File:** `app/compare/rbee-vs-ray-kserve/page.tsx`  
**Component:** `RbeeVsRayKservePage`  
**Purpose:** Competitive comparison vs Ray/KServe  
**Audience:** Enterprise evaluating Ray/KServe  
**Related PART doc:** PART_04_ComparisonPages.md

---

## `/features/*` Subroutes (6 pages)

All feature pages have dedicated components in `components/pages/`:

### `/features/multi-machine`
**Component:** `MultiMachinePage`  
**Purpose:** Multi-machine orchestration feature deep-dive  
**Related PART doc:** PART_03_FeaturesPage.md (mentions duplication)

---

### `/features/heterogeneous-hardware`
**Component:** `HeterogeneousHardwarePage`  
**Purpose:** Heterogeneous hardware support (CUDA + Metal + CPU)  
**Related PART doc:** PART_04_ComparisonPages.md

---

### `/features/ssh-deployment`
**Component:** Not found in components/pages/ (likely missing or named differently)  
**Purpose:** SSH deployment feature  
**Related PART doc:** PART_03_FeaturesPage.md

---

### `/features/rhai-scripting`
**Component:** `RhaiScriptingPage`  
**Purpose:** Rhai scripting feature  
**Related PART doc:** PART_03_FeaturesPage.md

---

### `/features/openai-compatible`
**Component:** `OpenAICompatiblePage`  
**Purpose:** OpenAI API compatibility  
**Related PART doc:** PART_03_FeaturesPage.md

---

### `/features/gdpr-compliance`
**Component:** `CompliancePage`  
**Purpose:** GDPR compliance feature  
**Related PART doc:** PART_03_FeaturesPage.md

---

## `/industries/*` Routes (10 pages)

**Note:** Massive duplication with top-level routes and use-cases

### `/industries/homelab`
**Component:** `HomelabPage`  
**Purpose:** Homelab use case  
**Audience:** Homelab enthusiasts  
**Related PART doc:** PART_06_UseCasePages.md

**Duplication Risk:** CRITICAL (80% overlap with HomePage per audit)

---

### `/industries/education`
**Component:** `EducationPage`  
**Purpose:** Education/teaching use case  
**Audience:** Educators  
**Related PART doc:** PART_05_AcademicPages.md

**Testimonials/Claims:**
- "Sarah Chen, CS Graduate → ML Engineer" (FABRICATED per audit)
- "500+ students taught" (NO UNIVERSITY NAMES)

**SEO Value:** 3/10 (worst in audit)  
**AI-Tone Severity:** CRITICAL (9/10)  
**Authority:** 1/10  
**Duplication Risk:** CRITICAL (70% overlap with ResearchPage)

---

### `/industries/research`
**Component:** `ResearchPage`  
**Purpose:** Research use case  
**Audience:** Researchers  
**Related PART doc:** PART_05_AcademicPages.md

**SEO Value:** 6/10  
**AI-Tone Severity:** MODERATE (6/10)  
**Duplication Risk:** HIGH (70% overlap with EducationPage)

**Note:** Audit recommends merging Education + Research → `/academic`

---

### `/industries/providers`
**Component:** `ProvidersPage`  
**Purpose:** GPU provider earnings  
**Audience:** GPU owners  
**Related PART doc:** PART_06_UseCasePages.md

**Duplication:** Same as `/gpu-providers`

---

### `/industries/developers`
**Component:** `DevelopersPage`  
**Purpose:** Developer use case  
**Audience:** Developers  

**Duplication:** Same as `/developers`

---

### `/industries/enterprise`
**Component:** `EnterprisePage`  
**Purpose:** Enterprise use case  
**Audience:** Enterprise  

**Duplication:** Same as `/enterprise`

---

### `/industries/startups`
**Component:** `StartupsPage`  
**Purpose:** Startup use case  
**Audience:** Startups  

---

### `/industries/devops`
**Component:** `DevOpsPage`  
**Purpose:** DevOps use case  
**Audience:** DevOps teams  

---

### `/industries/compliance`
**Component:** `CompliancePage`  
**Purpose:** Compliance use case  
**Audience:** Compliance officers  

**Duplication:** Likely overlaps with `/features/gdpr-compliance`

---

### `/industries/legal`
**Component:** `LegalPage`  
**Purpose:** Legal use case  
**Audience:** Legal teams  

**Duplication:** Confusing with `/legal` (legal docs)

---

## `/legal/*` Subroutes (3 pages)

### `/legal/privacy`
**Component:** `PrivacyPage`  
**Purpose:** Privacy policy  

---

### `/legal/terms`
**Component:** `TermsPage`  
**Purpose:** Terms of service  

---

## Summary Statistics

**Total Routes:** 32  
**Unique Components:** 27 (some routes share components)  
**Documented in PART docs:** ~9 pages  
**Undocumented in audit:** ~23 pages

**Duplication Patterns:**
1. **Top-level vs /industries:** 6 duplicates (developers, enterprise, providers, homelab, compliance, legal)
2. **HomePage sections:** Duplicated in features, use-cases, pricing
3. **Education vs Research:** 70% overlap (audit confirmed)
4. **Features overview vs feature pages:** Each feature has dedicated page + section on main features page

**Blocking Issues Found:**
- Fake testimonials: EducationPage, ProvidersPage
- Unsourced metrics: "87% GPU utilization", "500+ students", "€180K+ paid"
- AI-generic language: Especially EducationPage

**SEO Metadata Status:**
- ✅ Home: Full metadata + JSON-LD
- ✅ Pricing: Full metadata + JSON-LD + OG + Twitter
- ❌ Most other pages: No metadata exports found (need verification)

---

## Recommended Actions (from audit)

1. **DELETE duplicate routes:**
   - Merge `/industries/education` + `/industries/research` → `/academic`
   - Remove `/industries/homelab` or merge into `/use-cases`
   - Consolidate `/industries/*` duplicates with top-level pages

2. **FIX content issues:**
   - Remove fake testimonials
   - Source or delete unsourced metrics
   - Purge AI-generic language

3. **ADD SEO metadata:**
   - All pages missing metadata exports
   - Need title, description, keywords, JSON-LD

4. **CLARIFY structure:**
   - Choose between `/industries/*` OR top-level pages, not both
   - Create clear hierarchy for features (overview + deep-dives)
