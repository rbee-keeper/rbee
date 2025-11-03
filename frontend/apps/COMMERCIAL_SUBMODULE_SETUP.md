# Commercial Site Private Submodule Setup

## Overview

The commercial site (`frontend/apps/commercial`) is now a **private git submodule** to shield it from public repository access while maintaining pnpm workspace integration.

## 🚨 CRITICAL: The CMS Architecture Problem

### The Issue

The **PageProps files ARE the CMS** for the commercial site. They contain all marketing copy, CTAs, pricing, testimonials, and content configuration. These files were originally in the public `rbee-ui` package at:

```
frontend/packages/rbee-ui/src/pages/
├── HomePage/HomePageProps.tsx          ← Marketing copy, CTAs, hero content
├── EnterprisePage/EnterprisePageProps.tsx  ← Enterprise messaging
├── PricingPage/PricingPageProps.tsx    ← Pricing tiers, features
├── ProvidersPage/ProvidersPageProps.tsx    ← GPU provider messaging
└── [15 more pages...]                  ← All commercial content
```

**Problem:** This means all commercial messaging, pricing strategy, and marketing copy is **publicly visible** in the open-source repo. Competitors can see:
- Pricing strategy
- Target audience messaging
- Feature prioritization
- Marketing experiments
- A/B test variations

### The Solution

**Move all pages to the private commercial submodule:**

```
frontend/apps/commercial/components/pages/
├── HomePage/
│   ├── HomePage.tsx              ← React component (imports from @rbee/ui)
│   ├── HomePageProps.tsx         ← 🔒 PRIVATE marketing content
│   └── HomePage.stories.tsx      ← Storybook stories (optional)
├── EnterprisePage/
│   ├── EnterprisePage.tsx
│   ├── EnterprisePageProps.tsx   ← 🔒 PRIVATE
│   └── EnterprisePage.stories.tsx
└── [16 more pages...]
```

**What stays public in `rbee-ui`:**
- ✅ Templates (HeroTemplate, CTATemplate, etc.) - Generic, reusable
- ✅ Molecules (CodeBlock, TerminalWindow, etc.) - UI components
- ✅ Atoms (Button, Badge, etc.) - Design system
- ❌ Pages and PageProps - **MOVED TO PRIVATE**

### Why This Matters

The PageProps files are **not just configuration** - they are:
1. **Marketing strategy** - Messaging, positioning, value props
2. **Pricing intelligence** - Tier structure, feature gating
3. **Competitive advantage** - How we differentiate from Ollama, Runpod, etc.
4. **Content experiments** - A/B test variations, messaging tests

By moving them to a private submodule, we protect commercial strategy while keeping the UI library open-source.

## Initial Setup (One-time)

### 1. Create Private Repository

✅ **DONE** - Repository created at:
- SSH: `git@github.com:veighnsche/rbee-commercial-private.git`
- HTTPS: `https://github.com/veighnsche/rbee-commercial-private`

### 2. Migrate Pages from rbee-ui to Commercial

**CRITICAL STEP:** Move all page components and their props from the public package to the private submodule.

```bash
cd /home/vince/Projects/llama-orch

# Create pages directory in commercial site
mkdir -p frontend/apps/commercial/components/pages

# Copy all 18 page directories from rbee-ui to commercial
cp -r frontend/packages/rbee-ui/src/pages/* frontend/apps/commercial/components/pages/

# Verify the copy
ls frontend/apps/commercial/components/pages/
# Should show: CommunityPage, CompliancePage, DevOpsPage, DevelopersPage, 
#              EducationPage, EnterprisePage, FeaturesPage, HomePage, 
#              HomelabPage, LegalPage, PricingPage, PrivacyPage, 
#              ProvidersPage, ResearchPage, SecurityPage, StartupsPage, 
#              TermsPage, UseCasesPage, index.ts
```

**What gets migrated:**
- ✅ All `*Page.tsx` files (React components)
- ✅ All `*PageProps.tsx` files (🔒 **THE CMS** - marketing content)
- ✅ All `*Page.stories.tsx` files (Storybook stories)
- ✅ `index.ts` (exports)

**Total files:** ~60 files across 18 page directories

### 3. Update Import Paths in Commercial Site

After copying, update imports in the commercial Next.js app:

```bash
cd frontend/apps/commercial

# Find all imports from @rbee/ui/pages and update them
# Example: Change this in app/page.tsx or other route files:
# FROM: import { HomePage } from '@rbee/ui/pages'
# TO:   import { HomePage } from '@/components/pages/HomePage'
```

### 4. Initialize Commercial Site as Git Repo

```bash
cd /home/vince/Projects/llama-orch/frontend/apps/commercial

# Initialize as separate git repo
git init
git add .
git commit -m "Initial commit: Commercial site with migrated pages

- Migrated 18 page directories from rbee-ui
- PageProps files now private (CMS content)
- Commercial Next.js app structure
- Total: ~60 page component files"

# Add private remote
git remote add origin git@github.com:veighnsche/rbee-commercial-private.git
git branch -M main
git push -u origin main
```

### 5. Clean Up rbee-ui Package (Remove Pages)

After migrating pages to commercial, remove them from the public package:

```bash
cd /home/vince/Projects/llama-orch

# Remove pages directory from rbee-ui (now private in commercial)
rm -rf frontend/packages/rbee-ui/src/pages

# Update rbee-ui exports (remove pages export)
# Edit frontend/packages/rbee-ui/src/index.ts and remove:
# export * from './pages'

# Commit the cleanup
git add frontend/packages/rbee-ui
git commit -m "Remove pages from rbee-ui (migrated to private commercial submodule)

Pages are now in frontend/apps/commercial/components/pages
This keeps marketing content and CMS private while UI library stays open-source"
```

**What remains in rbee-ui (public):**
- ✅ `src/atoms/` - Design system primitives
- ✅ `src/molecules/` - Composite UI components
- ✅ `src/templates/` - Generic page templates
- ✅ `src/icons/` - Icon components
- ✅ `src/assets/` - Public assets
- ❌ `src/pages/` - **REMOVED** (now private)

### 6. Update .gitmodules URL

✅ **DONE** - Already updated to `git@github.com:veighnsche/rbee-commercial-private.git`

### 7. Remove from Main Repo and Add as Submodule

```bash
cd /home/vince/Projects/llama-orch

# Remove commercial directory from main repo tracking
git rm -r --cached frontend/apps/commercial

# Add as submodule (use your actual private repo URL)
git submodule add git@github.com:YOUR_USERNAME/rbee-commercial-private.git frontend/apps/commercial

# Commit the changes
git add .gitignore .gitmodules frontend/apps/commercial
git commit -m "Move commercial site to private submodule"
```

## For Other Developers (Cloning)

### Fresh Clone

```bash
# Clone main repo with submodules
git clone --recurse-submodules git@github.com:YOUR_USERNAME/llama-orch.git

# OR if already cloned without submodules
git submodule update --init --recursive
```

### Existing Clone

```bash
cd /home/vince/Projects/llama-orch

# Initialize and fetch submodule
git submodule update --init frontend/apps/commercial
```

## Working with the Submodule

### Update Commercial Site (Marketing Content)

```bash
cd frontend/apps/commercial

# Example: Update pricing page content (THE CMS)
# Edit components/pages/PricingPage/PricingPageProps.tsx
# Change pricing tiers, features, CTAs, etc.

git add components/pages/PricingPage/PricingPageProps.tsx
git commit -m "Update pricing: Add enterprise tier, adjust messaging"
git push origin main

# Go back to main repo and update submodule reference
cd /home/vince/Projects/llama-orch
git add frontend/apps/commercial
git commit -m "Update commercial submodule: New pricing structure"
git push
```

**Common CMS Updates:**
- 📝 Marketing copy changes → Edit `*PageProps.tsx` files
- 💰 Pricing updates → Edit `PricingPageProps.tsx`
- 🎯 A/B test variations → Create new props files
- 🚀 Feature launches → Update `FeaturesPageProps.tsx`
- 📊 Testimonials → Update `HomePageProps.tsx` or testimonial sections

### Pull Latest Commercial Changes

```bash
cd /home/vince/Projects/llama-orch

# Update submodule to latest commit
git submodule update --remote frontend/apps/commercial

# Commit the updated reference
git add frontend/apps/commercial
git commit -m "Update commercial submodule to latest"
```

## pnpm Workspace Integration

**No changes needed!** The `pnpm-workspace.yaml` still references `frontend/apps/commercial`, and pnpm will work seamlessly with the submodule.

```yaml
packages:
  - frontend/apps/commercial  # ✅ Still works as submodule
  - frontend/apps/user-docs
  # ... other packages
```

## Security Benefits

1. **Private Access Control**: Only users with access to the private repo can see the commercial site
2. **Public Repo Clean**: Main llama-orch repo remains open-source without commercial code
3. **Separate History**: Commercial site has its own git history
4. **Access Management**: Control who can view/edit commercial site via private repo permissions

## Troubleshooting

### Submodule Not Initialized

```bash
git submodule update --init frontend/apps/commercial
```

### Permission Denied

Ensure you have SSH access to the private repository:

```bash
ssh -T git@github.com
# Should show: Hi YOUR_USERNAME! You've successfully authenticated...
```

### pnpm Can't Find Package

```bash
# Ensure submodule is initialized
git submodule update --init

# Reinstall dependencies
pnpm install
```

## Architecture Overview

### Before (Public Pages Problem)

```
llama-orch/ (PUBLIC REPO)
├── frontend/
│   ├── packages/
│   │   └── rbee-ui/
│   │       └── src/
│   │           ├── pages/              ← 🚨 PROBLEM: Public CMS
│   │           │   ├── HomePage/
│   │           │   │   └── HomePageProps.tsx  ← Pricing, messaging PUBLIC
│   │           │   └── [17 more pages...]
│   │           ├── templates/          ← ✅ Generic, reusable (stays public)
│   │           ├── molecules/          ← ✅ UI components (stays public)
│   │           └── atoms/              ← ✅ Design system (stays public)
│   └── apps/
│       └── commercial/                 ← Next.js app (also public)
```

### After (Private Submodule Solution)

```
llama-orch/ (PUBLIC REPO)
├── .gitmodules                         ← Points to private repo
├── .gitignore                          ← Ignores frontend/apps/commercial/
├── frontend/
│   ├── packages/
│   │   └── rbee-ui/                    ← ✅ PUBLIC: UI library only
│   │       └── src/
│   │           ├── templates/          ← Generic templates
│   │           ├── molecules/          ← UI components
│   │           ├── atoms/              ← Design system
│   │           └── (no pages/)         ← REMOVED
│   └── apps/
│       └── commercial/                 ← 🔒 PRIVATE SUBMODULE
│           └── components/
│               └── pages/              ← 🔒 CMS lives here now
│                   ├── HomePage/
│                   │   ├── HomePage.tsx           ← Component
│                   │   └── HomePageProps.tsx      ← 🔒 PRIVATE content
│                   └── [17 more pages...]

rbee-commercial-private/ (PRIVATE REPO)
└── (same structure as frontend/apps/commercial/)
```

### File Structure in Private Submodule

```
frontend/apps/commercial/
├── app/                                ← Next.js 14 App Router
│   ├── page.tsx                        ← Homepage route
│   ├── pricing/page.tsx                ← Pricing route
│   ├── enterprise/page.tsx             ← Enterprise route
│   └── [other routes...]
├── components/
│   ├── pages/                          ← 🔒 THE CMS (migrated from rbee-ui)
│   │   ├── HomePage/
│   │   │   ├── HomePage.tsx            ← React component
│   │   │   ├── HomePageProps.tsx       ← 🔒 Marketing content
│   │   │   └── HomePage.stories.tsx    ← Storybook (optional)
│   │   ├── PricingPage/
│   │   │   ├── PricingPage.tsx
│   │   │   ├── PricingPageProps.tsx    ← 🔒 Pricing strategy
│   │   │   └── PricingPage.stories.tsx
│   │   └── [16 more pages...]
│   └── providers/                      ← App-specific components
├── package.json                        ← Dependencies
└── next.config.ts                      ← Next.js config
```

## Notes

- The commercial site is **gitignored** in the main repo (see `.gitignore`)
- The submodule reference is tracked in `.gitmodules`
- Changes to commercial site must be committed in **both** repos:
  1. Inside `frontend/apps/commercial` (the submodule)
  2. In the main repo (to update the submodule reference)
- **PageProps files are the CMS** - they contain all marketing content
- UI library (`rbee-ui`) remains open-source and generic
- Commercial content is private and protected from competitors

## Git History

**Decision:** Starting with a **clean git history** for the private repo.

**Why:** The pages were originally in the public `rbee-ui` package, so their git history is already public. Moving them with history would:
- ❌ Expose historical pricing experiments
- ❌ Show A/B test variations
- ❌ Reveal marketing strategy evolution

**Clean start benefits:**
- ✅ No historical baggage
- ✅ Fresh commit history for commercial content
- ✅ Easier to manage going forward
- ✅ No risk of accidentally exposing old strategies

The initial commit will be: "Initial commit: Commercial site with migrated pages"
