# Branch Naming Rules

**Created by:** TEAM-451  
**Enforced via:** GitHub branch protection + rulesets

---

## 🌳 Branch Structure

### Protected Branches (Permanent)

**`production`**
- Production-ready code
- Protected (requires PR + CI + approval)
- Only accepts merges from `development`
- Triggers releases on merge

**`development`**
- Default branch
- All work happens here
- Free pushing allowed
- Must pass CI before merging to production

---

## 📛 Branch Naming Convention

### Format

```
<type>/<team>-<description>
```

### Types

- `feat/` - New features
- `fix/` - Bug fixes
- `refactor/` - Code refactoring
- `docs/` - Documentation changes
- `test/` - Test additions/changes
- `chore/` - Maintenance tasks
- `release/` - Release preparation

### Examples

✅ **Good:**
```
feat/team-451-release-system
fix/team-450-build-errors
refactor/team-449-xtask-cleanup
docs/team-451-api-documentation
chore/team-451-cleanup-branches
```

❌ **Bad:**
```
commercial-site-updates        # No type prefix
mac-compat                     # No team number
stakeholder-story              # No type, no team
fix-bug                        # No team number
team-451                       # No type, no description
```

---

## 🔒 GitHub Branch Protection Rules

### Setup via GitHub UI

**Go to:** Settings → Branches → Add branch protection rule

### Rule 1: Protect `production`

**Branch name pattern:** `production`

**Settings:**
- ✅ Require a pull request before merging
  - ✅ Require approvals: 1
  - ✅ Dismiss stale pull request approvals when new commits are pushed
- ✅ Require status checks to pass before merging
  - ✅ Require branches to be up to date before merging
  - Status checks: `build`, `test`, `clippy`
- ✅ Require conversation resolution before merging
- ✅ Do not allow bypassing the above settings
- ✅ Restrict who can push to matching branches
  - Only: Administrators

### Rule 2: Protect `development`

**Branch name pattern:** `development`

**Settings:**
- ✅ Require status checks to pass before merging
  - Status checks: `build`, `test`
- ❌ Do NOT require pull requests (allow direct push)
- ❌ Do NOT restrict who can push

---

## 🎯 GitHub Branch Rulesets (Naming Enforcement)

**Note:** Rulesets are a newer GitHub feature that can enforce naming patterns.

### Setup via GitHub UI

**Go to:** Settings → Rules → Rulesets → New ruleset

### Ruleset: Enforce Branch Naming

**Target branches:** All branches except `production` and `development`

**Rules:**
1. **Restrict creations**
   - ✅ Require a pull request before merging
   
2. **Restrict updates**
   - Pattern must match: `^(feat|fix|refactor|docs|test|chore|release)/team-[0-9]+-[a-z0-9-]+$`

3. **Restrict deletions**
   - ✅ Prevent deletion of branches matching pattern

**Bypass list:**
- Administrators (for emergency fixes)

---

## 📋 Workflow

### Creating a New Branch

```bash
# From development
git checkout development
git pull origin development

# Create feature branch
git checkout -b feat/team-451-new-feature

# Work on feature
git add .
git commit -m "feat: implement new feature"
git push origin feat/team-451-new-feature

# When done, merge back to development
git checkout development
git merge feat/team-451-new-feature
git push origin development

# Delete feature branch
git branch -d feat/team-451-new-feature
git push origin --delete feat/team-451-new-feature
```

### Releasing to Production

```bash
# From development (after all features merged)
git checkout development

# Bump version
cargo xtask release --tier main --type minor

# Commit
git add .
git commit -m "chore: release main v0.2.0"
git push origin development

# Create PR to production
gh pr create \
  --base production \
  --head development \
  --title "Release rbee v0.2.0" \
  --body "Release notes..."

# After approval, merge PR
# GitHub Actions will build and release
```

---

## 🧹 Branch Cleanup

### Automatic Cleanup

**GitHub Setting:** Settings → General → Pull Requests
- ✅ Automatically delete head branches

### Manual Cleanup

```bash
# List merged branches
git branch --merged development | grep -v "development\|production\|main"

# Delete merged branches
git branch --merged development | grep -v "development\|production\|main" | xargs git branch -d

# Delete remote branches
git push origin --delete <branch-name>
```

### Cleanup Script

```bash
# Run the cleanup script
./scripts/cleanup-branches.sh
```

---

## 🚫 Forbidden Branches

These branch names are **NOT ALLOWED**:

- `master` - Use `production` instead
- `main` - Use `development` instead
- `dev` - Use `development` instead
- `prod` - Use `production` instead
- `staging` - We don't use staging
- `test` - Use `test/team-XXX-description` instead
- Any branch without team number
- Any branch without type prefix

---

## 📊 Branch Lifecycle

```
development (default)
    ↓
feat/team-451-new-feature (created)
    ↓
work, commit, push
    ↓
merge to development
    ↓
delete feat/team-451-new-feature
    ↓
development (updated)
    ↓
PR to production
    ↓
production (released)
```

---

## 🔧 Enforcement

### Via GitHub

1. **Branch protection rules** - Prevent direct push to production
2. **Branch rulesets** - Enforce naming patterns
3. **Required status checks** - CI must pass
4. **Required reviews** - At least 1 approval for production

### Via Git Hooks (Optional)

Create `.git/hooks/pre-push`:

```bash
#!/bin/bash
# Validate branch name before push

BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Skip validation for protected branches
if [[ "$BRANCH" == "development" || "$BRANCH" == "production" ]]; then
    exit 0
fi

# Validate branch name format
if ! [[ "$BRANCH" =~ ^(feat|fix|refactor|docs|test|chore|release)/team-[0-9]+-[a-z0-9-]+$ ]]; then
    echo "❌ Invalid branch name: $BRANCH"
    echo ""
    echo "Branch name must follow format: <type>/team-<number>-<description>"
    echo ""
    echo "Valid types: feat, fix, refactor, docs, test, chore, release"
    echo "Example: feat/team-451-release-system"
    exit 1
fi

exit 0
```

---

## 📝 Summary

**Permanent Branches:**
- `production` - Protected, releases only
- `development` - Default, free pushing

**Feature Branches:**
- Format: `<type>/team-<number>-<description>`
- Created from `development`
- Merged back to `development`
- Deleted after merge

**Releases:**
- PR from `development` to `production`
- Requires approval + CI
- Triggers automatic release

**Cleanup:**
- Delete branches after merge
- Run `./scripts/cleanup-branches.sh` periodically
- Enable auto-delete on GitHub

---

**Status:** Ready to implement  
**Next:** Run `./scripts/cleanup-branches.sh` to clean up old branches
