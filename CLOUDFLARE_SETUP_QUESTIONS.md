# Cloudflare Setup - Information Needed

**Created by:** TEAM-451  
**Status:** Waiting for your answers

---

## 📋 Please Answer These Questions

### 1. Cloudflare Account

**Do you have a Cloudflare account?**
- [ ] Yes
- [ ] No (I'll help you create one)

**If yes, what's the account email?**
```
Answer: ___________________________
```

---

### 2. Projects/Sites

**Do these projects already exist in Cloudflare?**
- [ ] Yes, all of them
- [ ] Yes, some of them
- [ ] No, none exist yet

**If they exist, what are the project names?**
```
Commercial site: ___________________________
Marketplace: ___________________________
User docs: ___________________________
Worker catalog: ___________________________
```

---

### 3. Domains

**What domains do you want to use?**

**Commercial site:**
```
Domain: ___________________________ (e.g., rbee.dev, commercial.rbee.dev)
```

**Marketplace:**
```
Domain: ___________________________ (e.g., marketplace.rbee.dev)
```

**User documentation:**
```
Domain: ___________________________ (e.g., docs.rbee.dev)
```

**Worker catalog API:**
```
Domain: ___________________________ (e.g., api.rbee.dev, workers.rbee.dev)
```

**Do you own these domains?**
- [ ] Yes, they're in Cloudflare
- [ ] Yes, but not in Cloudflare yet
- [ ] No, I need to buy them

---

### 4. Deployment Preference

**How do you want to deploy?**

**Option A: Manual (You control when)**
- You run: `wrangler deploy`
- Deploys when you say so
- Simple, predictable

**Option B: Automated (Git-based)**
- Push to production branch → auto-deploy
- Cloudflare watches GitHub
- Hands-off after setup

**Your choice:**
- [ ] Manual (I'll run wrangler deploy)
- [ ] Automated (Git push deploys)

---

### 5. Current Setup

**Do you have wrangler CLI installed?**
- [ ] Yes
- [ ] No

**If yes, are you logged in?**
```bash
# Check with:
wrangler whoami
```
- [ ] Yes, logged in
- [ ] No, not logged in

---

### 6. Build Configuration

**Where do you want to build?**

**Option A: Build on blep, deploy from blep**
- Install wrangler on blep
- Build and deploy from blep

**Option B: Build on mac, deploy from mac**
- Use mac for everything
- Consistent environment

**Option C: Build on blep, deploy from mac**
- Build on blep (faster)
- Deploy from mac (where wrangler is)

**Your choice:**
- [ ] Build and deploy from blep
- [ ] Build and deploy from mac
- [ ] Build on blep, deploy from mac

---

### 7. Environment Variables / Secrets

**Do your apps need any secrets?**
- API keys
- Database URLs
- Auth tokens
- etc.

**List them here:**
```
App: commercial
Secrets needed:
- ___________________________
- ___________________________

App: marketplace
Secrets needed:
- ___________________________
- ___________________________

App: user-docs
Secrets needed:
- ___________________________
- ___________________________

App: worker-catalog
Secrets needed:
- ___________________________
- ___________________________
```

---

## 🎯 Once You Answer These...

I'll create:

1. **Exact deployment commands** for your setup
2. **wrangler.toml configs** for each app
3. **Environment setup script** (if needed)
4. **Step-by-step deployment guide**
5. **Troubleshooting guide**

---

## 📝 Example Answers (for reference)

```
1. Cloudflare Account: yes, email: you@example.com
2. Projects exist: no
3. Domains:
   - Commercial: rbee.dev
   - Marketplace: marketplace.rbee.dev
   - Docs: docs.rbee.dev
   - Worker: api.rbee.dev
4. Deployment: Manual (I'll run wrangler deploy)
5. Wrangler: Not installed
6. Build location: Build and deploy from blep
7. Secrets: None for now
```

---

## 🚀 Quick Start (If You Want to Explore Now)

```bash
# Install wrangler
pnpm add -g wrangler

# Login to Cloudflare
wrangler login

# Check your account
wrangler whoami

# List your projects
wrangler pages project list

# List your workers
wrangler deployments list
```

---

**Fill this out and I'll create your exact deployment setup!** 📝
