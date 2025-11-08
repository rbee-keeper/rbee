# Manual Release Guide

**Created by:** TEAM-451  
**Workflow:** Manual (you control everything)

---

## 🎯 Release Workflow

### 1. Bump Version

```bash
# On blep
cd ~/Projects/rbee

# Interactive version bump
cargo xtask release

# Or specify tier and type
cargo xtask release --tier main --type minor
```

### 2. Commit and Push

```bash
# Commit the version changes
git add .
git commit -m "chore: release main v0.2.0"
git push origin development
```

### 3. Build on Mac

```bash
# SSH to mac and build
ssh mac << 'EOF'
cd ~/Projects/rbee
git pull origin development

# Build release binaries
cargo build --release

# Package binaries
cd target/release
tar -czf rbee-macos-arm64-v0.2.0.tar.gz \
  rbee-keeper \
  queen-rbee \
  rbee-hive

# Show what we built
ls -lh rbee-macos-arm64-v0.2.0.tar.gz
EOF
```

### 4. Download Binaries

```bash
# Download from mac to blep
scp mac:~/Projects/rbee/target/release/rbee-macos-arm64-v0.2.0.tar.gz .
```

### 5. Build on Blep (Linux)

```bash
# Build Linux binaries
cargo build --release

# Package binaries
cd target/release
tar -czf rbee-linux-x86_64-v0.2.0.tar.gz \
  rbee-keeper \
  queen-rbee \
  rbee-hive

cd ../..
```

### 6. Create GitHub Release

```bash
# Create release with both binaries
gh release create v0.2.0 \
  target/release/rbee-linux-x86_64-v0.2.0.tar.gz \
  rbee-macos-arm64-v0.2.0.tar.gz \
  --title "Release v0.2.0" \
  --notes "Release notes here"
```

---

## 📦 Cloudflare Publishing

### Frontend Apps (Cloudflare Pages)

**Apps to deploy:**
- `@rbee/commercial` → Commercial site
- `@rbee/marketplace` → Marketplace
- `@rbee/user-docs` → Documentation

**Questions I need answered:**

1. **Do you have a Cloudflare account?**
   - Yes/No
   - If yes, what's the account email?

2. **Are these projects already created in Cloudflare?**
   - Yes/No
   - If yes, what are the project names?

3. **What domains do you want to use?**
   - Commercial: `rbee.dev`? `commercial.rbee.dev`?
   - Marketplace: `marketplace.rbee.dev`?
   - Docs: `docs.rbee.dev`?

4. **Do you have API tokens?**
   - Yes/No
   - If no, I'll help you create them

5. **Manual or automated deployment?**
   - Manual: You run `wrangler deploy` yourself
   - Automated: Git push triggers deploy

### Worker Catalog (Cloudflare Workers)

**App:** `@rbee/global-worker-catalog`

**Questions:**

1. **What domain/subdomain?**
   - `api.rbee.dev`? `workers.rbee.dev`?

2. **Manual or automated?**
   - Manual: You run `wrangler deploy`
   - Automated: Git push triggers deploy

---

## 🔧 Manual Cloudflare Deployment (Once I Know Your Setup)

### Option 1: Manual Deploy (Simplest)

```bash
# Install wrangler globally
pnpm add -g wrangler

# Login to Cloudflare
wrangler login

# Deploy commercial site
cd frontend/apps/commercial
pnpm build
wrangler pages deploy dist --project-name=rbee-commercial

# Deploy marketplace
cd ../marketplace
pnpm build
wrangler pages deploy dist --project-name=rbee-marketplace

# Deploy docs
cd ../user-docs
pnpm build
wrangler pages deploy dist --project-name=rbee-docs

# Deploy worker catalog
cd ../../../bin/80-hono-worker-catalog
pnpm deploy
```

### Option 2: Automated (GitHub Integration)

Cloudflare can watch your GitHub repo and auto-deploy on push.

**Setup:**
1. Go to Cloudflare Pages dashboard
2. Connect to GitHub
3. Select repo: `rbee-keeper/rbee`
4. Configure build settings per app
5. Push to production → auto-deploy

---

## 📋 Complete Release Checklist

### Rust Binaries

- [ ] Run `cargo xtask release --tier main --type X`
- [ ] Commit and push to development
- [ ] SSH to mac and build: `cargo build --release`
- [ ] Package mac binaries: `tar -czf ...`
- [ ] Download to blep: `scp mac:...`
- [ ] Build on blep: `cargo build --release`
- [ ] Package blep binaries: `tar -czf ...`
- [ ] Create GitHub release: `gh release create ...`
- [ ] Upload both tarballs

### Frontend Apps (After Cloudflare Setup)

- [ ] Run `cargo xtask release --tier frontend --type X`
- [ ] Commit and push
- [ ] Deploy commercial: `wrangler pages deploy ...`
- [ ] Deploy marketplace: `wrangler pages deploy ...`
- [ ] Deploy docs: `wrangler pages deploy ...`
- [ ] Deploy worker: `wrangler deploy`

### Workers (Independent)

- [ ] Run `cargo xtask release --tier llm-worker --type X`
- [ ] Build worker: `cargo build --release --package llm-worker-rbee`
- [ ] Package and upload to GitHub release

---

## 🎯 Simplified Workflow (What You Actually Do)

### For Rust Binaries:

```bash
# 1. Bump version
cargo xtask release --tier main --type minor

# 2. Build on mac
ssh mac "cd ~/Projects/rbee && git pull && cargo build --release && cd target/release && tar -czf rbee-macos-arm64.tar.gz rbee-keeper queen-rbee rbee-hive"

# 3. Download
scp mac:~/Projects/rbee/target/release/rbee-macos-arm64.tar.gz .

# 4. Build on blep
cargo build --release
cd target/release && tar -czf rbee-linux-x86_64.tar.gz rbee-keeper queen-rbee rbee-hive && cd ../..

# 5. Release
gh release create v0.2.0 target/release/rbee-linux-x86_64.tar.gz rbee-macos-arm64.tar.gz
```

### For Frontend (Once Setup):

```bash
# 1. Bump version
cargo xtask release --tier frontend --type minor

# 2. Deploy
cd frontend/apps/commercial && pnpm build && wrangler pages deploy dist
cd ../marketplace && pnpm build && wrangler pages deploy dist
cd ../user-docs && pnpm build && wrangler pages deploy dist
```

---

## ❓ Tell Me About Your Cloudflare Setup

**Please answer these questions:**

1. Cloudflare account email: ________________
2. Do projects exist? (yes/no): ________________
3. Desired domains:
   - Commercial: ________________
   - Marketplace: ________________
   - Docs: ________________
   - Worker API: ________________
4. Deployment preference:
   - [ ] Manual (I run wrangler deploy)
   - [ ] Automated (Git push deploys)
5. Do you have wrangler installed? (yes/no): ________________

**Once I know this, I'll create the exact deployment commands you need!**

---

## 📚 Related Docs

- **Version Management:** `cargo xtask release --help`
- **Tier Configs:** `.version-tiers/*.toml`
- **Setup Complete:** `SETUP_COMPLETE.md`

---

**Status:** Ready for manual releases!  
**Next:** Answer Cloudflare questions so I can set that up
