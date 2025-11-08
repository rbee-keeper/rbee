# TEAM-457: Cloudflare Deployment Quick Guide

## 🚀 Deploy to Production (Zero Config)

All projects have production defaults built-in. Just build and deploy:

### Commercial Site
```bash
cd frontend/apps/commercial
pnpm build
wrangler pages deploy .open-next/assets --project-name rbee-commercial
```

### Marketplace
```bash
cd frontend/apps/marketplace
pnpm build
wrangler pages deploy .open-next/assets --project-name rbee-marketplace
```

### User Docs
```bash
cd frontend/apps/user-docs
pnpm build
wrangler pages deploy .open-next/assets --project-name rbee-user-docs
```

### Hono Worker Catalog
```bash
cd bin/80-hono-worker-catalog
wrangler deploy
```

---

## 🧪 Deploy to Preview

Test changes before production:

```bash
# Build with preview environment
wrangler pages deploy .open-next/assets --env preview

# Or for Hono worker
wrangler deploy --env preview
```

**Preview URLs:**
- Commercial: `https://preview.rbee.dev`
- Marketplace: `https://marketplace-preview.rbee.dev`

---

## 💻 Local Development

### Next.js Apps (Commercial, Marketplace, User Docs)

```bash
# 1. Copy env template (first time only)
cp .env.local.example .env.local

# 2. Start dev server
pnpm dev
```

**Ports:**
- Commercial: `http://localhost:3000`
- Marketplace: `http://localhost:3001`
- User Docs: `http://localhost:3002`

### Hono Worker Catalog

```bash
# Start with development environment
wrangler dev --env development
```

**Port:** `http://localhost:8787`

---

## 🔧 Environment Variables

### Production (Default)
All projects use these by default:
- `NEXT_PUBLIC_SITE_URL`: `https://rbee.dev`
- `NEXT_PUBLIC_MARKETPLACE_URL`: `https://marketplace.rbee.dev`
- `NEXT_PUBLIC_DOCS_URL`: `https://docs.rbee.dev`
- `NEXT_PUBLIC_GITHUB_URL`: `https://github.com/veighnsche/llama-orch`

### Override for Local Testing

Edit `.env.local`:
```bash
# Test with local marketplace
NEXT_PUBLIC_MARKETPLACE_URL=http://localhost:3001
NEXT_PUBLIC_SITE_URL=http://localhost:3000
```

---

## 📋 Pre-Deployment Checklist

- [ ] Run `pnpm build` successfully
- [ ] No TypeScript errors
- [ ] Test navigation links locally
- [ ] Verify environment variables are correct
- [ ] Check `wrangler.jsonc` has correct project name

---

## 🐛 Troubleshooting

### Build Fails
```bash
# Clean and rebuild
rm -rf .next .open-next node_modules
pnpm install
pnpm build
```

### Wrong URLs in Production
Check `wrangler.jsonc` vars section. Production values should be there.

### Local Dev Not Using .env.local
Make sure file is named exactly `.env.local` (not `.env.local.txt` or similar).

### Wrangler Command Not Found
```bash
pnpm install -g wrangler
# or
npx wrangler [command]
```

---

## 🔧 Generate TypeScript Types

After modifying any `wrangler.jsonc` file, regenerate types:

```bash
# From project root
./scripts/generate-cloudflare-types.sh

# Or manually for a specific project
cd frontend/apps/commercial
pnpm dlx wrangler types
```

This creates `worker-configuration.d.ts` with type-safe environment variables.

---

## 📚 More Info

See `TEAM_457_CLOUDFLARE_READY_COMPLETE.md` for comprehensive documentation.
