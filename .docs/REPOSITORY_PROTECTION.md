# Repository Protection Guide

**Created by:** TEAM-451  
**Purpose:** Protect rbee repository from trolls, accidents, and unauthorized changes

## 🛡️ Protection Layers

### 1. Branch Protection Rules
### 2. Repository Settings
### 3. Collaborator Permissions
### 4. Required Status Checks
### 5. Code Review Requirements

---

## 1. Branch Protection Rules

### Production Branch (Strictest)

**Branch:** `production`

**Required Settings:**
- ✅ Require pull request before merging
  - ✅ Require approvals: **2** (for production)
  - ✅ Dismiss stale reviews when new commits are pushed
  - ✅ Require review from Code Owners (if CODEOWNERS file exists)
  - ✅ Require approval of most recent reviewable push
- ✅ Require status checks to pass before merging
  - ✅ Require branches to be up to date before merging
  - Required checks: `build`, `test`, `clippy`, `frontend-build`
- ✅ Require conversation resolution before merging
- ✅ Require signed commits (recommended)
- ✅ Require linear history (prevents merge commits)
- ❌ Do not allow bypassing the above settings
- ❌ Do not allow force pushes
- ❌ Do not allow deletions
- ✅ Lock branch (read-only, optional for extra protection)

### Development Branch (Moderate)

**Branch:** `development`

**Required Settings:**
- ✅ Require pull request before merging
  - ✅ Require approvals: **1**
  - ✅ Dismiss stale reviews
- ✅ Require status checks to pass
  - ❌ Do not require branches to be up to date (faster iteration)
  - Required checks: `build`, `test`
- ✅ Require conversation resolution
- ✅ Allow force pushes (for cleanup/rebasing)
- ❌ Do not allow deletions

### Main Branch (Legacy/Backup)

**Branch:** `main`

**Required Settings:**
- ✅ Require pull request before merging
- ✅ Require approvals: **1**
- ❌ Do not allow force pushes
- ❌ Do not allow deletions

---

## 2. Repository Settings

### General Settings

**Location:** `https://github.com/rbee-keeper/rbee/settings`

#### Features to Enable
- ✅ **Issues** - Bug tracking
- ✅ **Discussions** - Community Q&A
- ✅ **Preserve this repository** - Archive for posterity
- ❌ **Wikis** - Use docs/ instead
- ❌ **Projects** - Use GitHub Projects separately if needed

#### Pull Requests
- ✅ Allow merge commits
- ✅ Allow squash merging (default)
- ✅ Allow rebase merging
- ✅ Always suggest updating pull request branches
- ✅ Automatically delete head branches (cleanup)

#### Pushes
- ❌ Limit who can push to matching branches (use branch protection instead)

#### Danger Zone
- ✅ Make repository private (if needed)
- ❌ Do NOT enable "Allow merge commits" without squash option
- ❌ Do NOT disable branch protection

---

## 3. Collaborator Permissions

### Permission Levels

**Admin** (You + trusted maintainers only)
- Can change repository settings
- Can manage branch protection
- Can delete repository
- **Limit to 1-2 people**

**Maintain** (Core team)
- Can manage issues/PRs
- Cannot change settings
- Cannot bypass branch protection

**Write** (Regular contributors)
- Can push to non-protected branches
- Can create PRs
- Cannot merge to protected branches

**Triage** (Community moderators)
- Can manage issues/PRs
- Cannot push code

**Read** (Public/external contributors)
- Can view and fork
- Can create issues/PRs from forks

### Recommended Setup
```
You (vince)           → Admin
Core team (1-2)       → Maintain
Regular contributors  → Write
Community helpers     → Triage
Everyone else         → Read (public repo) or None (private)
```

---

## 4. Required Status Checks

### GitHub Actions Workflows

Create these required checks in `.github/workflows/`:

#### `ci.yml` - Continuous Integration
```yaml
name: CI

on:
  pull_request:
    branches: [development, production, main]
  push:
    branches: [development, production]

jobs:
  build:
    name: Build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo build --all-targets
      
  test:
    name: Test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test --all
      
  clippy:
    name: Clippy
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy
      - run: cargo clippy --all-targets --all-features -- -D warnings
      
  frontend-build:
    name: Frontend Build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v2
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: pnpm
      - run: pnpm install
      - run: pnpm run build
```

#### Mark as Required
1. Go to branch protection settings
2. Under "Require status checks to pass before merging"
3. Search for: `build`, `test`, `clippy`, `frontend-build`
4. Select all and save

---

## 5. Code Review Requirements

### CODEOWNERS File

Create `.github/CODEOWNERS`:

```
# Global owners (review all changes)
* @vince

# Rust backend
/bin/**/*.rs @vince
/contracts/**/*.rs @vince
Cargo.toml @vince

# Frontend
/frontend/**/*.ts @vince
/frontend/**/*.tsx @vince
package.json @vince

# CI/CD
/.github/workflows/*.yml @vince

# Documentation
/docs/**/*.md @vince
/.docs/**/*.md @vince

# Critical files (require extra review)
/scripts/*.sh @vince
/.windsurf/rules/*.md @vince
```

This ensures you're notified of all changes and can require your review.

---

## 🚀 Quick Setup Commands

### Step 1: Authenticate
```bash
gh auth login
```

### Step 2: Create Branches
```bash
./scripts/setup-github-branches.sh
```

### Step 3: Configure Protection (Requires Admin)
```bash
./scripts/setup-github-branches.sh --protect
```

### Step 4: Manual Configuration (Web UI)

**For Production Branch:**
```bash
# Open in browser
gh browse --settings

# Navigate to:
# Settings → Branches → Add rule → Branch name: production
```

Then configure:
1. ✅ Require pull request (2 approvals)
2. ✅ Require status checks: `build`, `test`, `clippy`, `frontend-build`
3. ✅ Require conversation resolution
4. ✅ Require signed commits
5. ✅ Require linear history
6. ❌ Do not allow force pushes
7. ❌ Do not allow deletions
8. ❌ Do not allow bypassing

**For Development Branch:**
```bash
# Same as above but:
# - Only 1 approval required
# - Allow force pushes
# - Don't require branches to be up to date
```

---

## 🔒 Additional Protection Measures

### 1. Enable Two-Factor Authentication (2FA)
**Critical for admins!**
```
GitHub → Settings → Password and authentication → Two-factor authentication
```

### 2. Signed Commits
```bash
# Generate GPG key
gpg --full-generate-key

# List keys
gpg --list-secret-keys --keyid-format=long

# Configure git
git config --global user.signingkey <KEY_ID>
git config --global commit.gpgsign true

# Add to GitHub
gpg --armor --export <KEY_ID>
# Paste into: GitHub → Settings → SSH and GPG keys → New GPG key
```

### 3. Dependabot Security Updates
Enable in repository settings:
```
Settings → Code security and analysis → Dependabot
✅ Dependabot alerts
✅ Dependabot security updates
✅ Dependabot version updates
```

### 4. Secret Scanning
```
Settings → Code security and analysis → Secret scanning
✅ Secret scanning
✅ Push protection
```

### 5. Private Vulnerability Reporting
```
Settings → Code security and analysis
✅ Private vulnerability reporting
```

---

## 🚨 Troll Protection Checklist

- [ ] Branch protection enabled on `production` (2 approvals)
- [ ] Branch protection enabled on `development` (1 approval)
- [ ] Required status checks configured
- [ ] Force push disabled on protected branches
- [ ] Branch deletion disabled on protected branches
- [ ] CODEOWNERS file created
- [ ] Only trusted users have Admin access
- [ ] Two-factor authentication enabled for all admins
- [ ] Dependabot enabled
- [ ] Secret scanning enabled
- [ ] Issue/PR templates created (optional but helpful)
- [ ] Contribution guidelines documented (CONTRIBUTING.md)

---

## 📋 Issue/PR Templates (Optional)

### `.github/ISSUE_TEMPLATE/bug_report.yml`
```yaml
name: Bug Report
description: File a bug report
labels: ["bug"]
body:
  - type: textarea
    id: what-happened
    attributes:
      label: What happened?
      description: Describe the bug
    validations:
      required: true
      
  - type: textarea
    id: reproduction
    attributes:
      label: Steps to reproduce
      description: How can we reproduce this?
    validations:
      required: true
```

### `.github/PULL_REQUEST_TEMPLATE.md`
```markdown
## Description
<!-- Describe your changes -->

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Checklist
- [ ] Tests pass locally
- [ ] Clippy passes
- [ ] Documentation updated
- [ ] Follows engineering rules
```

---

## 🔍 Monitoring & Auditing

### View Protection Status
```bash
# Check branch protection
gh api repos/rbee-keeper/rbee/branches/production/protection

# List collaborators
gh api repos/rbee-keeper/rbee/collaborators

# View recent events
gh api repos/rbee-keeper/rbee/events
```

### Audit Log
**For organizations:**
```
Organization → Settings → Audit log
```

**For personal repos:**
- Limited audit log in repository Insights → Traffic

---

## 📚 References

- [GitHub Branch Protection](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches)
- [CODEOWNERS Syntax](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)
- [Required Status Checks](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches#require-status-checks-before-merging)
- [Signed Commits](https://docs.github.com/en/authentication/managing-commit-signature-verification)

---

## 🎯 Summary

**Minimum Protection (Do This Now):**
1. Run `./scripts/setup-github-branches.sh --protect`
2. Configure production branch: 2 approvals, no force push
3. Configure development branch: 1 approval, allow force push
4. Enable 2FA for your account

**Recommended Protection (Do This Soon):**
5. Create CODEOWNERS file
6. Set up required status checks (CI workflow)
7. Enable Dependabot
8. Enable secret scanning

**Advanced Protection (Optional):**
9. Require signed commits
10. Create issue/PR templates
11. Set up branch rulesets (new GitHub feature)
12. Configure security policies
