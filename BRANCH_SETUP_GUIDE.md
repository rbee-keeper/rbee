# Branch Setup Guide - DO THIS NOW

**Created by:** TEAM-451  
**Priority:** CRITICAL

---

## 🎯 Goal

- Delete `main` branch
- Use `development` as default
- Use `production` for releases
- Clean up old branches
- Enforce naming rules

---

## ✅ Step-by-Step Instructions

### Step 1: Cleanup Old Branches (5 minutes)

```bash
# Run the cleanup script
./scripts/cleanup-branches.sh
```

This will delete:
- `commercial-site-updates`
- `fix/team-117-ambiguous-steps`
- `fix/team-122-panics-final`
- `mac-compat`
- `stakeholder-story`

---

### Step 2: Switch to Development (1 minute)

```bash
# Switch to development
git checkout development

# Make sure it's up to date
git pull origin development

# Merge any work from main (if needed)
git merge main
git push origin development
```

---

### Step 3: Set Development as Default on GitHub (2 minutes)

**Manual steps:**

1. Go to: https://github.com/rbee-keeper/rbee/settings
2. Click **"Branches"** in left sidebar
3. Under "Default branch", click the switch icon
4. Select **`development`**
5. Click **"Update"**
6. Confirm the change

---

### Step 4: Delete Main Branch (1 minute)

**ONLY AFTER Step 3 is complete!**

```bash
# Delete local main
git branch -d main

# Delete remote main
git push origin --delete main
```

---

### Step 5: Setup Branch Protection (5 minutes)

#### 5.1 Protect Production Branch

**Go to:** Settings → Branches → Add branch protection rule

**Branch name pattern:** `production`

**Enable:**
- ✅ Require a pull request before merging
  - ✅ Require approvals: 1
  - ✅ Dismiss stale pull request approvals when new commits are pushed
- ✅ Require status checks to pass before merging
  - ✅ Require branches to be up to date before merging
- ✅ Require conversation resolution before merging
- ✅ Do not allow bypassing the above settings
- ✅ Restrict who can push to matching branches
  - Only: Administrators

**Click:** "Create" or "Save changes"

#### 5.2 Protect Development Branch (Light Protection)

**Branch name pattern:** `development`

**Enable:**
- ✅ Require status checks to pass before merging
  - Status checks: (add when CI is setup)

**Leave DISABLED:**
- ❌ Require pull requests (allow direct push)
- ❌ Restrict who can push

**Click:** "Create" or "Save changes"

---

### Step 6: Setup Branch Naming Rules (Optional, 5 minutes)

**Go to:** Settings → Rules → Rulesets → New ruleset

**Name:** "Enforce Branch Naming"

**Target branches:** All branches

**Bypass list:** Administrators

**Rules:**
1. **Branch name pattern:**
   ```
   ^(development|production|feat|fix|refactor|docs|test|chore|release)/.*$
   ```

2. **Require pattern for branch creation:**
   - Pattern: `^(feat|fix|refactor|docs|test|chore|release)/team-[0-9]+-[a-z0-9-]+$`
   - Exempt: `development`, `production`

**Click:** "Create ruleset"

---

## 📋 Verification Checklist

After completing all steps:

- [ ] Old branches deleted
- [ ] On `development` branch
- [ ] `development` is default on GitHub
- [ ] `main` branch deleted (local and remote)
- [ ] `production` branch protected (requires PR + approval)
- [ ] `development` branch has light protection (CI checks only)
- [ ] Branch naming rules configured (optional)

---

## 🎯 Result

**Before:**
```
main (default, unprotected)
development (unprotected)
production (unprotected)
+ 5 old feature branches
```

**After:**
```
development (default, light protection)
production (protected, requires PR)
```

---

## 📚 Documentation

**Branch naming rules:** `.docs/BRANCH_NAMING_RULES.md`

**Branch naming format:**
```
<type>/team-<number>-<description>

Examples:
feat/team-451-release-system
fix/team-450-build-errors
docs/team-451-api-docs
```

---

## 🚀 Next Steps

After branch setup is complete:

1. **Fix tier configs** (see `IMMEDIATE_ACTION_PLAN.md`)
2. **Setup Cloudflare deployments**
3. **First release!**

---

**Status:** Ready to execute  
**Time:** ~15 minutes total
