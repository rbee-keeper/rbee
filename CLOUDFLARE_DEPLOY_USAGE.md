# Cloudflare Deployment - Quick Usage

**Created by:** TEAM-451

---

## 🚀 Individual Deployment Commands

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

## 🎯 Summary

**4 individual commands, deploy in order:**

```bash
cargo xtask deploy --app worker       # gwc.rbee.dev
cargo xtask deploy --app commercial   # rbee.dev
cargo xtask deploy --app marketplace  # marketplace.rbee.dev
cargo xtask deploy --app docs         # docs.rbee.dev
```

**All run on mac, all automatic!** 🚀
