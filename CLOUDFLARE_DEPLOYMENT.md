# Cloudflare Deployment Guide

**Created by:** TEAM-451  
**Your Setup:** Build and deploy from mac using xtask

---

## 🎯 Your Configuration

**Domains:**
- Commercial: `rbee.dev`
- Marketplace: `marketplace.rbee.dev`
- Docs: `docs.rbee.dev`
- Worker Catalog: `gwc.rbee.dev`

**Account:** `vincepaul.liem@gmail.com` (ID: `cf772d0960afaac63a91ba755590e524`)  
**Deployment:** Manual via `cargo xtask deploy`  
**Build Location:** mac

---

## 🚀 Quick Start

### Deploy Everything

```bash
# On mac
cargo xtask deploy --all

# Or deploy specific apps
cargo xtask deploy --app commercial
cargo xtask deploy --app marketplace
cargo xtask deploy --app docs
cargo xtask deploy --app worker
```

---

## 📋 Environment Variables Needed

### Commercial Site (`rbee.dev`)

**Public variables (build-time):**
```bash
NEXT_PUBLIC_MARKETPLACE_URL=https://marketplace.rbee.dev
NEXT_PUBLIC_SITE_URL=https://rbee.dev
NEXT_PUBLIC_GITHUB_URL=https://github.com/rbee-keeper/rbee
NEXT_PUBLIC_DOCS_URL=https://docs.rbee.dev
NEXT_PUBLIC_LEGAL_EMAIL=legal@rbee.dev
NEXT_PUBLIC_SUPPORT_EMAIL=support@rbee.dev
```

**No secrets needed!** All variables are public (NEXT_PUBLIC_*)

### Marketplace (`marketplace.rbee.dev`)

**Variables needed:**
```bash
MARKETPLACE_API_URL=https://gwc.rbee.dev  # Points to worker catalog
NEXT_DISABLE_DEVTOOLS=1
```

**Question:** Do you have a separate marketplace API, or does it use the worker catalog?
- If worker catalog: Use `https://gwc.rbee.dev`
- If separate API: What's the URL?

### User Docs (`docs.rbee.dev`)

**Variables needed:**
```bash
NEXT_PUBLIC_SITE_URL=https://docs.rbee.dev
NEXT_PUBLIC_GITHUB_URL=https://github.com/rbee-keeper/rbee
```

**No secrets needed!**

### Worker Catalog (`gwc.rbee.dev`)

**No environment variables found in code!**

This is a Hono worker - typically needs:
- Database connection (if using D1/KV)
- API keys (if calling external services)

**Question:** What does the worker catalog do?
- Serve static worker metadata?
- Connect to a database?
- Call external APIs?

---

## 🔧 Manual Deployment (Until xtask is Ready)

### 1. Commercial Site

```bash
ssh mac << 'EOF'
cd ~/Projects/rbee/frontend/apps/commercial

# Create .env.local
cat > .env.local << 'ENVEOF'
NEXT_PUBLIC_MARKETPLACE_URL=https://marketplace.rbee.dev
NEXT_PUBLIC_SITE_URL=https://rbee.dev
NEXT_PUBLIC_GITHUB_URL=https://github.com/rbee-keeper/rbee
NEXT_PUBLIC_DOCS_URL=https://docs.rbee.dev
NEXT_PUBLIC_LEGAL_EMAIL=legal@rbee.dev
NEXT_PUBLIC_SUPPORT_EMAIL=support@rbee.dev
ENVEOF

# Build
pnpm build

# Deploy to Cloudflare Pages
wrangler pages deploy .next --project-name=rbee-commercial --branch=production

# Set custom domain
wrangler pages domain add rbee-commercial rbee.dev
EOF
```

### 2. Marketplace

```bash
ssh mac << 'EOF'
cd ~/Projects/rbee/frontend/apps/marketplace

# Create .env.local
cat > .env.local << 'ENVEOF'
MARKETPLACE_API_URL=https://gwc.rbee.dev
NEXT_DISABLE_DEVTOOLS=1
ENVEOF

# Build
pnpm build

# Deploy
wrangler pages deploy .next --project-name=rbee-marketplace --branch=production

# Set custom domain
wrangler pages domain add rbee-marketplace marketplace.rbee.dev
EOF
```

### 3. User Docs

```bash
ssh mac << 'EOF'
cd ~/Projects/rbee/frontend/apps/user-docs

# Create .env.local
cat > .env.local << 'ENVEOF'
NEXT_PUBLIC_SITE_URL=https://docs.rbee.dev
NEXT_PUBLIC_GITHUB_URL=https://github.com/rbee-keeper/rbee
ENVEOF

# Build
pnpm build

# Deploy
wrangler pages deploy .next --project-name=rbee-docs --branch=production

# Set custom domain
wrangler pages domain add rbee-docs docs.rbee.dev
EOF
```

### 4. Worker Catalog

```bash
ssh mac << 'EOF'
cd ~/Projects/rbee/bin/80-hono-worker-catalog

# Create wrangler.toml
cat > wrangler.toml << 'WRANGLEREOF'
name = "rbee-worker-catalog"
main = "src/index.ts"
compatibility_date = "2024-11-01"

[env.production]
name = "rbee-worker-catalog"
routes = [
  { pattern = "gwc.rbee.dev", custom_domain = true }
]
WRANGLEREOF

# Deploy
pnpm deploy
EOF
```

---

## 🎯 xtask Deploy Implementation

I'll create `cargo xtask deploy` that does all of this for you!

**Features:**
- Deploy all apps or specific ones
- Automatic environment variable setup
- Version bumping integration
- Rollback support
- Deployment verification

**Usage:**
```bash
# Deploy everything
cargo xtask deploy --all

# Deploy specific app
cargo xtask deploy --app commercial

# Deploy with version bump
cargo xtask release --tier frontend --type minor
cargo xtask deploy --all

# Dry run
cargo xtask deploy --all --dry-run
```

---

## 📊 Deployment Checklist

### First-Time Setup

- [ ] Create Cloudflare Pages projects
  ```bash
  ssh mac "cd ~/Projects/rbee/frontend/apps/commercial && wrangler pages project create rbee-commercial"
  ssh mac "cd ~/Projects/rbee/frontend/apps/marketplace && wrangler pages project create rbee-marketplace"
  ssh mac "cd ~/Projects/rbee/frontend/apps/user-docs && wrangler pages project create rbee-docs"
  ```

- [ ] Set custom domains
  ```bash
  ssh mac "wrangler pages domain add rbee-commercial rbee.dev"
  ssh mac "wrangler pages domain add rbee-marketplace marketplace.rbee.dev"
  ssh mac "wrangler pages domain add rbee-docs docs.rbee.dev"
  ```

- [ ] Deploy worker catalog
  ```bash
  ssh mac "cd ~/Projects/rbee/bin/80-hono-worker-catalog && pnpm deploy"
  ```

- [ ] Verify DNS is pointing to Cloudflare
  ```bash
  dig rbee.dev
  dig marketplace.rbee.dev
  dig docs.rbee.dev
  dig gwc.rbee.dev
  ```

### Regular Deployment

- [ ] Bump version: `cargo xtask release --tier frontend --type X`
- [ ] Commit: `git add . && git commit -m "chore: release frontend vX.Y.Z"`
- [ ] Push: `git push origin development`
- [ ] Deploy: `cargo xtask deploy --all` (once implemented)

---

## 🐛 Troubleshooting

### Build fails

```bash
# Check Node/pnpm versions
ssh mac "node --version && pnpm --version"

# Clean and rebuild
ssh mac "cd ~/Projects/rbee && pnpm clean && pnpm install && pnpm build"
```

### Deployment fails

```bash
# Check wrangler auth
ssh mac "wrangler whoami"

# Check project exists
ssh mac "wrangler pages project list"

# View deployment logs
ssh mac "wrangler pages deployment list --project-name=rbee-commercial"
```

### Domain not working

```bash
# Check DNS
dig rbee.dev

# Check Cloudflare Pages domain
ssh mac "wrangler pages domain list --project-name=rbee-commercial"

# Re-add domain
ssh mac "wrangler pages domain add rbee-commercial rbee.dev"
```

---

## ❓ Questions I Still Need Answered

### 1. Marketplace API

**Question:** Does marketplace use the worker catalog API, or do you have a separate backend?

**Options:**
- A) Use worker catalog: `MARKETPLACE_API_URL=https://gwc.rbee.dev`
- B) Separate API: What's the URL?
- C) No API needed: Remove the variable

### 2. Worker Catalog Functionality

**Question:** What does the worker catalog do?

**If it needs:**
- **Database:** I'll add D1/KV configuration
- **External APIs:** I'll add secrets management
- **Just serves static data:** No additional config needed

### 3. Email Addresses

**Question:** Do these email addresses exist?
- `legal@rbee.dev`
- `support@rbee.dev`

**If not:** I can set up Cloudflare Email Routing (free)

---

## 🎯 Next Steps

1. **Answer the questions above**
2. **I'll implement `cargo xtask deploy`**
3. **Test first deployment manually**
4. **Use xtask for future deployments**

---

**Ready to deploy once you answer the questions!** 🚀
