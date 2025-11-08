# Cloudflare Setup - Information Needed

**Created by:** TEAM-451  
**Status:** Waiting for your answers

---

## 📋 Please Answer These Questions

### 1. Cloudflare Account

**Do you have a Cloudflare account?**
- [X] Yes
- [ ] No (I'll help you create one)

**If yes, what's the account email?**
```
Answer: vincepaul.liem@gmail.com
```

---

### 2. Projects/Sites

**Do these projects already exist in Cloudflare?**
- [ ] Yes, all of them
- [ ] Yes, some of them
- [X] No, none exist yet

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
Domain: rbee.dev

**Marketplace:**
```
Domain: marketplace.rbee.dev
```

**User documentation:**
```
Domain: docs.rbee.dev
```

**Worker catalog API:**
```
Domain: gwc.rbee.dev
```

**Do you own these domains?**
- [X] Yes, they're in Cloudflare
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
- [X] Use a deployment helper script through xtask

---

### 5. Current Setup

**Do you have wrangler CLI installed?**
- [X] Yes
❯ wrangler -v

 ⛅️ wrangler 4.46.0
───────────────────
- [ ] No

**If yes, are you logged in?**
```bash
# Check with:
wrangler whoami
```
- [X] Yes, logged in
❯ wrangler whoami

 ⛅️ wrangler 4.46.0
───────────────────
Getting User settings...
👋 You are logged in with an OAuth Token, associated with the email vincepaul.liem@gmail.com.
┌────────────────────────────────────┬──────────────────────────────────┐
│ Account Name                       │ Account ID                       │
├────────────────────────────────────┼──────────────────────────────────┤
│ Junebonnet@hotmail.nl's Account    │ bce75e6d72016186da22d710ef811e77 │ <- ABSOLUTELY NO IDEA WHERE THIS COMES FROM, please ignore
├────────────────────────────────────┼──────────────────────────────────┤
│ Vincepaul.liem@gmail.com's Account │ cf772d0960afaac63a91ba755590e524 │
└────────────────────────────────────┴──────────────────────────────────┘
🔓 Token Permissions:
Scope (Access)
- account (read)
- user (read)
- workers (write)
- workers_kv (write)
- workers_routes (write)
- workers_scripts (write)
- workers_tail (read)
- d1 (write)
- pages (write)
- zone (read)
- ssl_certs (write)
- ai (write)
- queues (write)
- pipelines (write)
- secrets_store (write)
- containers (write)
- cloudchamber (write)
- connectivity (admin)
- offline_access 
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
- [X] Build and deploy from mac
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
- ___________________________ PLEASE HELP FINDING THESE
- ___________________________

App: marketplace
Secrets needed:
- ___________________________ PLEASE HELP FINDING THESE
- ___________________________

App: user-docs
Secrets needed:
- ___________________________ PLEASE HELP FINDING THESE
- ___________________________

App: worker-catalog
Secrets needed:
- ___________________________ PLEASE HELP FINDING THESE
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
