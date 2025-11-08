# Quick Deploy Guide

**Purpose:** Quick reference for deploying apps  
**When confused:** Read this first!

---

## 🚀 Deploy Commands (NOT Release!)

### Deploy Worker Catalog (gwc.rbee.dev)
```bash
cargo xtask deploy --app worker
# or
cargo xtask deploy --app gwc
# or
cargo xtask deploy --app worker-catalog
```

### Deploy Other Apps
```bash
# Commercial site
cargo xtask deploy --app commercial

# Marketplace
cargo xtask deploy --app marketplace

# Docs
cargo xtask deploy --app docs
```

---

## 📦 Release Commands (Version Bumping)

### Release Frontend Apps
```bash
cargo xtask release
# Select: frontend (Frontend applications and Cloudflare Workers)
# Select bump type: patch/minor/major
```

### Release Main Binaries
```bash
cargo xtask release
# Select: main (User-facing binaries)
# Select bump type: patch/minor/major
```

---

## ❌ Common Mistakes

**Wrong:** `cargo xtask release` when you want to deploy  
**Right:** `cargo xtask deploy --app worker`

**Wrong:** Looking for "gwc" in release tiers  
**Right:** Worker catalog is in "frontend" tier (for version bumping)

---

## 🎯 Quick Decision Tree

**Want to deploy an app?**
→ `cargo xtask deploy --app <name>`

**Want to bump version?**
→ `cargo xtask release`

**Want to see all deploy options?**
→ `cargo xtask deploy --help`

---

## 📚 Full Documentation

- `CLOUDFLARE_DEPLOY_USAGE.md` - All deployment commands
- `MANUAL_RELEASE_GUIDE.md` - Release workflow
- `bin/80-hono-worker-catalog/PRODUCTION_MARKETPLACE_PLAN.md` - Worker catalog plan
