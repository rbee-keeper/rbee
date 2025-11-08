# IMMEDIATE ACTION PLAN - TEAM-451

**Created:** 2025-11-08  
**Priority:** CRITICAL - Do this BEFORE continuing with release work

---

## 🚨 STOP - You're Right, We Need to Fix This First

### Problem 1: Still on `main` branch
- You want: `development` (default) and `production` (protected)
- Reality: Still on `main`

### Problem 2: JavaScript packages incomplete
- Current: Only 2 JS packages (queen-rbee-sdk, rbee-hive-sdk)
- Missing: Frontend apps and Cloudflare Workers

### Problem 3: No deployment strategy for frontend
- Commercial site (Cloudflare Pages)
- Marketplace (Cloudflare Pages)
- User docs (Cloudflare Pages)
- Hono worker catalog (Cloudflare Workers)

---

## ✅ Step 1: Fix Branches (DO THIS NOW - 5 minutes)

### 1.1 Switch to development

```bash
cd /home/vince/Projects/rbee

# Switch to development
git checkout development

# Merge any uncommitted work from main
git merge main

# Push to ensure development is up to date
git push origin development
```

### 1.2 Set development as default branch

**On GitHub:**
1. Go to https://github.com/rbee-keeper/rbee/settings
2. Click "Branches" in left sidebar
3. Change default branch from `main` to `development`
4. Confirm the change

### 1.3 Delete main branch (AFTER setting development as default)

```bash
# Delete local main
git branch -d main

# Delete remote main (ONLY after changing default on GitHub)
git push origin --delete main
```

### 1.4 Setup branch protection

```bash
# Run the script we created
./scripts/configure-branch-protection.sh
```

**Result:**
- ✅ `development` - Default, free pushing
- ✅ `production` - Protected, requires PR + CI

---

## ✅ Step 2: Fix JavaScript Package List (10 minutes)

### Current `.version-tiers/main.toml`

```toml
[javascript]
packages = [
    "@rbee/queen-rbee-sdk",
    "@rbee/rbee-hive-sdk",
]
```

### What It Should Be

The SDKs are WASM bindings built during Rust compilation. They're not separate releases.

**Remove them from main tier** - they're built artifacts, not source packages.

### Create New Tier: `frontend.toml`

```toml
# .version-tiers/frontend.toml
name = "frontend"
description = "Frontend applications and Cloudflare Workers"

[rust]
crates = []
shared_crates = []

[javascript]
packages = [
    "@rbee/commercial",           # Cloudflare Pages
    "@rbee/marketplace",          # Cloudflare Pages
    "@rbee/user-docs",            # Cloudflare Pages
    "@rbee/hono-worker-catalog",  # Cloudflare Workers
]
```

---

## ✅ Step 3: Understand the Build/Deploy Flow

### Rust Binaries (Main Tier)
```
cargo build --release
  ├─> rbee-keeper (CLI)
  ├─> queen-rbee (daemon)
  ├─> rbee-hive (daemon)
  └─> WASM SDKs (built as part of queen/hive build)
       ├─> @rbee/queen-rbee-sdk (artifact)
       └─> @rbee/rbee-hive-sdk (artifact)
```

**Release:** GitHub Releases (binaries)

### Workers (Independent Tiers)
```
cargo build --release --features cuda
  └─> llm-worker-rbee-cuda
```

**Release:** GitHub Releases (binaries)

### Frontend Apps (New Tier)
```
pnpm build
  ├─> commercial (Cloudflare Pages)
  ├─> marketplace (Cloudflare Pages)
  ├─> user-docs (Cloudflare Pages)
  └─> hono-worker-catalog (Cloudflare Workers)
```

**Release:** Cloudflare (NOT GitHub Releases)

---

## ✅ Step 4: Deployment Strategy

### GitHub Releases (Binaries)
- **What:** Rust binaries (keeper, queen, hive, workers)
- **How:** GitHub Actions → GitHub Releases
- **When:** Merge to `production`

### Cloudflare Pages (Frontend Apps)
- **What:** Commercial, marketplace, user-docs
- **How:** Cloudflare Pages GitHub integration
- **When:** Merge to `production`
- **Config:** `wrangler.toml` in each app

### Cloudflare Workers (Hono Catalog)
- **What:** Worker catalog API
- **How:** `wrangler deploy`
- **When:** Merge to `production`
- **Config:** `bin/80-hono-worker-catalog/wrangler.toml`

---

## 🎯 Corrected Release Workflow

### Scenario 1: Main Release (Rust Binaries)

```bash
# 1. Bump version
cargo xtask release --tier main --type minor

# 2. Commit
git add .
git commit -m "chore: release main v0.2.0"
git push origin development

# 3. Create PR
gh pr create --base production --head development --title "Release rbee v0.2.0"

# 4. Merge → GitHub Actions builds binaries
# 5. GitHub Release created with binaries
```

### Scenario 2: Frontend Release

```bash
# 1. Bump version
cargo xtask release --tier frontend --type minor

# 2. Commit
git add .
git commit -m "chore: release frontend v2.2.0"
git push origin development

# 3. Create PR
gh pr create --base production --head development --title "Release frontend v2.2.0"

# 4. Merge → Cloudflare deploys automatically
```

### Scenario 3: Worker Release

```bash
# Same as before - independent
cargo xtask release --tier llm-worker --type patch
```

---

## 📋 Action Items (In Order)

### IMMEDIATE (Do Now)
- [ ] **1. Switch to development branch** (5 min)
- [ ] **2. Set development as default on GitHub** (2 min)
- [ ] **3. Delete main branch** (1 min)
- [ ] **4. Run branch protection script** (2 min)

### NEXT (After branches fixed)
- [ ] **5. Remove WASM SDKs from main.toml** (1 min)
- [ ] **6. Create frontend.toml tier** (5 min)
- [ ] **7. Test: `cargo xtask release --tier frontend --dry-run`** (1 min)

### LATER (Deployment Setup)
- [ ] **8. Setup Cloudflare Pages for commercial** (15 min)
- [ ] **9. Setup Cloudflare Pages for marketplace** (15 min)
- [ ] **10. Setup Cloudflare Pages for user-docs** (15 min)
- [ ] **11. Setup Cloudflare Workers for hono-catalog** (15 min)
- [ ] **12. Test full deployment workflow** (30 min)

---

## 🚨 What We Got Wrong

### Assumption 1: WASM SDKs are separate packages
**Wrong:** They're build artifacts from Rust compilation  
**Right:** They're generated during `cargo build`, not separate releases

### Assumption 2: All JS packages release together
**Wrong:** Frontend apps deploy to Cloudflare, not GitHub  
**Right:** Need separate deployment strategy for each platform

### Assumption 3: One release workflow for everything
**Wrong:** Different platforms need different workflows  
**Right:** 
- Rust binaries → GitHub Releases
- Frontend apps → Cloudflare Pages
- Workers → Cloudflare Workers

---

## 🎯 Corrected Architecture

```
rbee Release System
├─ Tier: main (Rust binaries)
│  ├─ GitHub Actions → Build
│  └─ GitHub Releases → Distribute
│
├─ Tier: llm-worker (Rust binary)
│  ├─ GitHub Actions → Build
│  └─ GitHub Releases → Distribute
│
├─ Tier: sd-worker (Rust binary)
│  ├─ GitHub Actions → Build
│  └─ GitHub Releases → Distribute
│
└─ Tier: frontend (JS apps)
   ├─ Cloudflare Pages → commercial, marketplace, docs
   └─ Cloudflare Workers → hono-catalog
```

---

## 💡 Next Steps

**RIGHT NOW:**
1. Fix branches (Steps 1.1-1.4)
2. Fix tier configs (Steps 5-7)

**THEN:**
1. Setup Cloudflare deployments (Steps 8-11)
2. Test everything (Step 12)

**FINALLY:**
1. First real release!

---

## 📞 Questions to Answer

Before continuing, we need to know:

1. **Cloudflare setup:**
   - Do you have Cloudflare account?
   - Do you have API tokens?
   - Are projects already created?

2. **Domain setup:**
   - What domains for commercial/marketplace/docs?
   - Are they already configured in Cloudflare?

3. **WASM SDKs:**
   - Are they published to npm?
   - Or just used internally?
   - Should they be versioned separately?

---

**Status:** PAUSED - Fix branches first, then we'll continue

**DO THIS NOW:** Steps 1.1-1.4 (branch setup)
