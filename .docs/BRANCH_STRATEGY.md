# Branch Strategy

**Created by:** TEAM-451  
**Date:** 2025-11-08

## Overview

This repository uses a three-branch strategy for development and releases:

```
main (legacy) ──┐
                ├──> development (default) ──> production (protected)
feature/* ──────┘
```

## Branches

### `development` (Default Branch)
- **Purpose:** Active development and integration
- **Protection:** Minimal (allows force push for cleanup)
- **Workflow:** All feature branches merge here first
- **CI/CD:** Runs tests on every push

### `production` (Protected)
- **Purpose:** Production-ready code
- **Protection:** Strict
  - Requires pull request reviews (1+ approvals)
  - Dismisses stale reviews
  - No force pushes
  - No deletions
  - Requires conversation resolution
- **Workflow:** Only accepts merges from `development` via PR
- **CI/CD:** Runs full test suite + deployment

### `main` (Legacy)
- **Purpose:** Original development branch
- **Status:** Can be kept for historical reference or deleted
- **Recommendation:** Migrate work to `development`

## Setup

### Prerequisites

```bash
# Install GitHub CLI (Arch Linux)
paru -S github-cli

# Authenticate
gh auth login
```

### Quick Setup

Run the automated setup script:

```bash
./scripts/setup-github-branches-simple.sh
```

This will:
1. Create `development` and `production` branches
2. Push them to GitHub
3. Set `development` as the default branch
4. Provide instructions for branch protection

### Manual Setup

If you prefer manual setup or need admin access:

1. **Create branches:**
   ```bash
   git checkout -b development
   git push -u origin development
   
   git checkout -b production
   git push -u origin production
   ```

2. **Set default branch:**
   ```bash
   gh repo edit rbee-keeper/rbee --default-branch development
   ```

3. **Configure branch protection (via GitHub UI):**
   - Go to: https://github.com/rbee-keeper/rbee/settings/branches
   - For `production`:
     - ✅ Require pull request before merging
     - ✅ Require approvals (1)
     - ✅ Dismiss stale reviews
     - ✅ Require conversation resolution
     - ✅ Do not allow bypassing settings
     - ❌ Allow force pushes
     - ❌ Allow deletions

## Workflow

### Feature Development

```bash
# Start from development
git checkout development
git pull origin development

# Create feature branch
git checkout -b feature/my-feature

# Work on feature
git add .
git commit -m "feat: add my feature"

# Push and create PR to development
git push -u origin feature/my-feature
gh pr create --base development --title "feat: add my feature"
```

### Release to Production

```bash
# Ensure development is ready
git checkout development
git pull origin development

# Run tests
cargo test --all
cargo clippy --all-targets --all-features -- -D warnings

# Create PR to production
gh pr create --base production --head development --title "Release: $(date +%Y-%m-%d)"

# After review and approval, merge via GitHub UI
```

### Hotfix

```bash
# Create hotfix from production
git checkout production
git pull origin production
git checkout -b hotfix/critical-bug

# Fix and test
git add .
git commit -m "fix: critical bug"

# PR to production
git push -u origin hotfix/critical-bug
gh pr create --base production --title "hotfix: critical bug"

# After merge, backport to development
git checkout development
git merge production
git push origin development
```

## CI/CD Integration

### Development Branch
- Runs on every push
- Fast feedback loop
- Can fail without blocking

### Production Branch
- Runs full test suite
- Requires all checks to pass
- Triggers deployment on merge

### Example GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [development, production]
  pull_request:
    branches: [development, production]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: cargo test --all
      
  clippy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run clippy
        run: cargo clippy --all-targets --all-features -- -D warnings
```

## Branch Protection via API

If you have admin access, you can use the GitHub API:

```bash
# Production branch protection
gh api repos/rbee-keeper/rbee/branches/production/protection \
  --method PUT \
  -f required_pull_request_reviews[required_approving_review_count]=1 \
  -f required_pull_request_reviews[dismiss_stale_reviews]=true \
  -f enforce_admins=true \
  -f allow_force_pushes=false \
  -f allow_deletions=false \
  -f required_conversation_resolution=true

# Development branch protection (lighter)
gh api repos/rbee-keeper/rbee/branches/development/protection \
  --method PUT \
  -f allow_force_pushes=true \
  -f allow_deletions=false
```

## Troubleshooting

### "Resource not accessible by integration"
- You need admin access to the repository
- Configure branch protection via GitHub UI instead

### "Branch not found"
- Ensure branches are pushed to remote: `git push -u origin <branch>`
- Check branch exists: `git branch -a`

### Authentication issues
- Re-authenticate: `gh auth login`
- Check status: `gh auth status`

## References

- [GitHub Branch Protection](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- [GitHub CLI Manual](https://cli.github.com/manual/)
- [Git Flow](https://nvie.com/posts/a-successful-git-branching-model/)
