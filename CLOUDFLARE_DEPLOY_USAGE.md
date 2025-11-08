# Deployment Commands - Quick Usage

**Created by:** TEAM-451

---

## 🎯 All 9 Deployment Commands

### Cloudflare (4 apps)
- `cargo xtask deploy --app worker` → gwc.rbee.dev
- `cargo xtask deploy --app commercial` → rbee.dev
- `cargo xtask deploy --app marketplace` → marketplace.rbee.dev
- `cargo xtask deploy --app docs` → docs.rbee.dev

### GitHub Releases (5 binaries)
- `cargo xtask deploy --app keeper` → rbee-keeper (macOS + Linux)
- `cargo xtask deploy --app queen` → queen-rbee (macOS + Linux)
- `cargo xtask deploy --app hive` → rbee-hive (macOS + Linux)
- `cargo xtask deploy --app llm-worker` → llm-worker-rbee (macOS + Linux)
- `cargo xtask deploy --app sd-worker` → sd-worker-rbee (macOS + Linux)

---

## 🚀 Cloudflare Deployment Commands

### 1. Deploy Worker Catalog (FIRST!)

```bash
# On mac
cargo xtask deploy --app worker

# Or use aliases
cargo xtask deploy --app gwc
cargo xtask deploy --app worker-catalog
```

**Deploys to:** `gwc.rbee.dev`  
**Why first?** Marketplace needs this API

---

### 2. Deploy Commercial Site

```bash
cargo xtask deploy --app commercial
```

**Deploys to:** `rbee.dev`

---

### 3. Deploy Marketplace

```bash
cargo xtask deploy --app marketplace
```

**Deploys to:** `marketplace.rbee.dev`  
**Requires:** Worker catalog deployed first

---

### 4. Deploy User Docs

```bash
cargo xtask deploy --app docs
# Or
cargo xtask deploy --app user-docs
```

**Deploys to:** `docs.rbee.dev`

---

## 🎯 Complete Deployment Workflow

### First Time Setup

```bash
# 1. Deploy worker catalog (marketplace needs this)
cargo xtask deploy --app worker

# 2. Deploy all frontend apps
cargo xtask deploy --app commercial
cargo xtask deploy --app marketplace
cargo xtask deploy --app docs
```

### Regular Updates

```bash
# 1. Bump version
cargo xtask release --tier frontend --type patch

# 2. Commit
git add .
git commit -m "chore: release frontend v0.1.1"
git push origin development

# 3. Deploy (on mac)
ssh mac << 'EOF'
cd ~/Projects/rbee
git pull origin development

# Deploy all apps
cargo xtask deploy --app worker
cargo xtask deploy --app commercial
cargo xtask deploy --app marketplace
cargo xtask deploy --app docs
EOF
```

---

## 🔍 Dry Run (Preview)

Test what will happen without actually deploying:

```bash
cargo xtask deploy --app worker --dry-run
cargo xtask deploy --app commercial --dry-run
cargo xtask deploy --app marketplace --dry-run
cargo xtask deploy --app docs --dry-run
```

---

## 📋 What Each Command Does

### Worker Catalog

1. Creates `wrangler.toml` (if missing)
2. Runs `pnpm deploy` in `bin/80-hono-worker-catalog`
3. Deploys to `gwc.rbee.dev`

### Commercial Site

1. Creates `.env.local` with production URLs
2. Runs `pnpm build`
3. Deploys `.next` to Cloudflare Pages
4. Custom domain: `rbee.dev`

### Marketplace

1. Creates `.env.local` with `MARKETPLACE_API_URL=https://gwc.rbee.dev`
2. Runs `pnpm build`
3. Deploys `.next` to Cloudflare Pages
4. Custom domain: `marketplace.rbee.dev`

### User Docs

1. Creates `.env.local` with production URLs
2. Runs `pnpm build`
3. Deploys `.next` to Cloudflare Pages
4. Custom domain: `docs.rbee.dev`

---

## ⚠️ Important Notes

### Deployment Order

**ALWAYS deploy worker catalog first!**

```
1. worker catalog (gwc.rbee.dev)
   ↓
2. marketplace (needs worker catalog API)
   ↓
3. commercial & docs (can be any order)
```

### Environment Variables

All environment variables are automatically created:
- ✅ Commercial: Public URLs and emails
- ✅ Marketplace: Points to `gwc.rbee.dev`
- ✅ Docs: Public URLs
- ✅ Worker: No env vars needed (serves static data)

### First Deployment

First time you deploy, you may need to setup custom domains:

```bash
# On mac
wrangler pages domain add rbee-commercial rbee.dev
wrangler pages domain add rbee-marketplace marketplace.rbee.dev
wrangler pages domain add rbee-docs docs.rbee.dev
```

The deploy commands will remind you if needed.

---

## 🐛 Troubleshooting

### Build fails

```bash
# Check Node/pnpm versions
ssh mac "node --version && pnpm --version"

# Clean and rebuild
ssh mac "cd ~/Projects/rbee && pnpm clean && pnpm install"
```

### Deployment fails

```bash
# Check wrangler auth
ssh mac "wrangler whoami"

# Check projects exist
ssh mac "wrangler pages project list"
```

### Wrong API URL

If marketplace can't reach worker catalog:

```bash
# Verify worker is deployed
curl https://gwc.rbee.dev/health

# Should return: {"status":"ok","service":"worker-catalog","version":"0.1.0"}
```

---

## 📊 Verification

After deployment, verify each site:

```bash
# Worker catalog
curl https://gwc.rbee.dev/health

# Commercial site
curl https://rbee.dev

# Marketplace
curl https://marketplace.rbee.dev

# Docs
curl https://docs.rbee.dev
```

---

## 🔨 Rust Binary Deployment Commands

### Deploy Individual Binaries

Each command builds on both mac (ARM64) and blep (x86_64), then uploads to GitHub Releases:

```bash
# rbee-keeper (main tier)
cargo xtask deploy --app keeper

# queen-rbee (main tier)
cargo xtask deploy --app queen

# rbee-hive (main tier)
cargo xtask deploy --app hive

# llm-worker-rbee (llm-worker tier)
cargo xtask deploy --app llm-worker

# sd-worker-rbee (sd-worker tier)
cargo xtask deploy --app sd-worker
```

### What Each Binary Deployment Does

1. **Gets version** from tier config (`.version-tiers/*.toml`)
2. **Builds on mac** via SSH: `cargo build --release --package <binary>`
3. **Packages mac binary**: `tar -czf <binary>-macos-arm64-<version>.tar.gz`
4. **Downloads** from mac to blep
5. **Builds on blep**: `cargo build --release --package <binary>`
6. **Packages blep binary**: `tar -czf <binary>-linux-x86_64-<version>.tar.gz`
7. **Creates GitHub release** (if doesn't exist)
8. **Uploads both tarballs** to release

### Complete Binary Release Workflow

```bash
# 1. Bump version
cargo xtask release --tier main --type minor

# 2. Commit
git add .
git commit -m "chore: release main v0.2.0"
git push origin development

# 3. Deploy all main tier binaries
cargo xtask deploy --app keeper
cargo xtask deploy --app queen
cargo xtask deploy --app hive

# 4. Deploy worker binaries (if updated)
cargo xtask release --tier llm-worker --type patch
cargo xtask deploy --app llm-worker

cargo xtask release --tier sd-worker --type patch
cargo xtask deploy --app sd-worker
```

### Download Released Binaries

```bash
# Download latest release
gh release download v0.2.0

# Download specific binary
gh release download v0.2.0 --pattern '*keeper*'

# Download only macOS binaries
gh release download v0.2.0 --pattern '*macos*'

# Download only Linux binaries
gh release download v0.2.0 --pattern '*linux*'
```

### Dry Run (Preview)

Test what will happen without actually deploying:

```bash
cargo xtask deploy --app keeper --dry-run
cargo xtask deploy --app queen --dry-run
cargo xtask deploy --app hive --dry-run
cargo xtask deploy --app llm-worker --dry-run
cargo xtask deploy --app sd-worker --dry-run
```

### Binary Deployment Troubleshooting

**Build fails on mac:**
```bash
# Check Rust version
ssh mac "rustc --version"

# Clean and rebuild
ssh mac "cd ~/Projects/rbee && cargo clean && cargo build --release --package rbee-keeper"
```

**Build fails on blep:**
```bash
# Check Rust version
rustc --version

# Clean and rebuild
cargo clean
cargo build --release --package rbee-keeper
```

**Release creation fails:**
```bash
# Check gh CLI auth
gh auth status

# Check if release already exists
gh release list

# View specific release
gh release view v0.2.0
```

**Upload fails:**
```bash
# Delete and recreate release
gh release delete v0.2.0 --yes
cargo xtask deploy --app keeper
```

---

## 🎯 Summary

**9 deployment commands total:**

### Cloudflare (4 apps)
```bash
cargo xtask deploy --app worker       # gwc.rbee.dev
cargo xtask deploy --app commercial   # rbee.dev
cargo xtask deploy --app marketplace  # marketplace.rbee.dev
cargo xtask deploy --app docs         # docs.rbee.dev
```

### GitHub Releases (5 binaries)
```bash
cargo xtask deploy --app keeper       # rbee-keeper (macOS + Linux)
cargo xtask deploy --app queen        # queen-rbee (macOS + Linux)
cargo xtask deploy --app hive         # rbee-hive (macOS + Linux)
cargo xtask deploy --app llm-worker   # llm-worker-rbee (macOS + Linux)
cargo xtask deploy --app sd-worker    # sd-worker-rbee (macOS + Linux)
```

**All automated, all individual!** 🚀
