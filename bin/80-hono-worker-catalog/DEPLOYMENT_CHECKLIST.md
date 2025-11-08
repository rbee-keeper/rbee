# Worker Catalog Deployment Checklist

**Purpose:** Checklist for deploying worker catalog specifically  
**When to use:** Before deploying `cargo xtask deploy --app worker`  
**Related docs:** `../../CLOUDFLARE_DEPLOY_USAGE.md` - All deployment commands

---

## 🤖 Automated Pre-Deployment (Deployment Gates)

**All these checks run automatically when you deploy!**

The deployment gates will automatically verify:

1. ✅ **TypeScript type check** - Code compiles
2. ✅ **Lint check** - Code quality standards
3. ✅ **Unit tests** - All tests pass (data validation + API tests)
4. ✅ **Build test** - Production build succeeds
5. ✅ **PKGBUILD validation** - All 16 package files exist
6. ✅ **Install script validation** - install.sh exists and is executable

**You don't need to run these manually!** Just run the deploy command and the gates will check everything.

---

## 👤 Manual Pre-Deployment Checks

These are the ONLY things you need to check manually:

### ✅ 1. Environment Variables

- [ ] `CLOUDFLARE_API_TOKEN` is set
  ```bash
  echo $CLOUDFLARE_API_TOKEN
  # Should show your API token
  ```

- [ ] `CLOUDFLARE_ACCOUNT_ID` is set (if needed)
  ```bash
  echo $CLOUDFLARE_ACCOUNT_ID
  # Should show your account ID
  ```

### ✅ 2. Wrangler Authentication

- [ ] Logged into wrangler
  ```bash
  wrangler whoami
  # Should show your Cloudflare account
  ```

---

## 🚀 Deployment Steps

### Step 1: Run Deployment Gates

```bash
# From project root
cargo xtask deploy --app worker --dry-run
```

**Expected output:**
```
🚦 Running deployment gates for worker...

📦 Worker Catalog Gates:
  1. TypeScript type check... ✓
  2. Lint check... ✓
  3. Unit tests... ✓
  4. Build test... ✓

✅ All deployment gates passed for worker
```

- [ ] All gates passed

### Step 2: Deploy to Cloudflare

```bash
# Actual deployment (no dry-run)
cargo xtask deploy --app worker
```

**Expected output:**
```
🚀 Deploying Worker Catalog to gwc.rbee.dev

⚠️  wrangler.toml not found, creating it...
✅ Created wrangler.toml

📦 Deploying to Cloudflare...
✅ Deployed successfully!
```

- [ ] Deployment succeeded
- [ ] No errors in output

### Step 3: Verify Deployment

Wait 30 seconds for DNS propagation, then test:

- [ ] Health check
  ```bash
  curl https://gwc.rbee.dev/health
  # Should return: {"status":"ok","service":"worker-catalog","version":"0.1.0"}
  ```

- [ ] List workers
  ```bash
  curl https://gwc.rbee.dev/workers
  # Should return 5 workers
  ```

- [ ] Get specific worker
  ```bash
  curl https://gwc.rbee.dev/workers/llm-worker-rbee-cpu
  # Should return worker details
  ```

- [ ] Install script
  ```bash
  curl https://gwc.rbee.dev/install.sh | head -20
  # Should return the install script (first 20 lines)
  ```

- [ ] Test from marketplace
  ```bash
  # If marketplace is deployed, check it can reach worker catalog
  curl https://marketplace.rbee.dev
  # Should load without errors
  ```

---

## 🔧 Post-Deployment Verification

### ✅ 1. DNS & SSL

- [ ] `gwc.rbee.dev` resolves correctly
  ```bash
  dig gwc.rbee.dev
  ```

- [ ] SSL certificate is valid
  ```bash
  curl -I https://gwc.rbee.dev
  # Should return: HTTP/2 200
  ```

### ✅ 2. CORS Headers

- [ ] CORS headers present
  ```bash
  curl -I https://gwc.rbee.dev/workers
  # Should include: access-control-allow-origin header
  ```

### ✅ 3. Performance

- [ ] Response time < 500ms
  ```bash
  curl -w "@-" -o /dev/null -s https://gwc.rbee.dev/workers <<'EOF'
  time_total: %{time_total}s
  EOF
  ```

### ✅ 4. Error Handling

- [ ] 404 returns proper error
  ```bash
  curl -i https://gwc.rbee.dev/workers/invalid
  # Should return: 404 with {"error":"Worker not found"}
  ```

---

## 🎯 Integration Tests

### Test with rbee-keeper

- [ ] List available workers
  ```bash
  rbee worker catalog
  # Should show 5 workers from gwc.rbee.dev
  ```

- [ ] Get worker details
  ```bash
  rbee worker get llm-worker-rbee-cpu
  # Should show worker details
  ```

### Test install script

- [ ] Install script works
  ```bash
  curl -fsSL https://gwc.rbee.dev/install.sh | head -50
  # Should show install script with proper headers
  ```

---

## 📊 Monitoring

### ✅ 1. Cloudflare Dashboard

- [ ] Check Cloudflare Workers dashboard
  - Go to: https://dash.cloudflare.com
  - Navigate to Workers & Pages
  - Find: `rbee-worker-catalog`
  - Verify: Status is "Active"

### ✅ 2. Analytics

- [ ] Check request count
  - Should see requests in analytics
  - No errors in logs

### ✅ 3. Logs

- [ ] Check for errors in Cloudflare logs
  - No 500 errors
  - No unhandled exceptions

---

## 🐛 Troubleshooting

### If deployment fails:

1. **Check wrangler auth:**
   ```bash
   wrangler whoami
   ```

2. **Check wrangler.toml:**
   ```bash
   cat bin/80-hono-worker-catalog/wrangler.toml
   ```

3. **Try manual deploy:**
   ```bash
   cd bin/80-hono-worker-catalog
   pnpm deploy
   ```

4. **Check Cloudflare dashboard:**
   - Look for deployment errors
   - Check worker logs

### If endpoints don't work:

1. **Check DNS:**
   ```bash
   dig gwc.rbee.dev
   ```

2. **Check SSL:**
   ```bash
   curl -I https://gwc.rbee.dev
   ```

3. **Check CORS:**
   ```bash
   curl -H "Origin: http://localhost:7823" -I https://gwc.rbee.dev/workers
   ```

4. **Check worker logs in Cloudflare dashboard**

---

## 📝 Rollback Plan

If something goes wrong:

### Option 1: Redeploy previous version

```bash
# From git history
git log --oneline bin/80-hono-worker-catalog
git checkout <previous-commit> bin/80-hono-worker-catalog
cargo xtask deploy --app worker
```

### Option 2: Manual rollback in Cloudflare

1. Go to Cloudflare dashboard
2. Navigate to Workers & Pages
3. Find `rbee-worker-catalog`
4. Click "Rollback" to previous deployment

---

## ✅ Final Checklist

- [ ] All pre-deployment checks passed
- [ ] Deployment succeeded
- [ ] All endpoints responding correctly
- [ ] DNS resolves correctly
- [ ] SSL certificate valid
- [ ] CORS headers present
- [ ] Performance acceptable (< 500ms)
- [ ] Error handling works
- [ ] Integration tests passed
- [ ] Monitoring shows no errors
- [ ] Marketplace can reach worker catalog
- [ ] Install script accessible

---

## 🎉 Success Criteria

**Deployment is successful when:**

1. ✅ `curl https://gwc.rbee.dev/health` returns `{"status":"ok"}`
2. ✅ `curl https://gwc.rbee.dev/workers` returns 5 workers
3. ✅ `curl https://gwc.rbee.dev/install.sh` returns install script
4. ✅ No errors in Cloudflare logs
5. ✅ Marketplace can fetch worker data

---

## 📞 Support

**If you encounter issues:**

1. Check Cloudflare dashboard for errors
2. Check deployment logs
3. Review this checklist
4. Check `DEPLOYMENT_GATES_SUMMARY.md` for gate details
5. Check `PKGBUILD_PLAN.md` for PKGBUILD structure

**Deployment command reference:**
```bash
# Dry run (test without deploying)
cargo xtask deploy --app worker --dry-run

# Actual deployment
cargo xtask deploy --app worker
```

---

**Last updated:** 2025-11-09  
**Version:** 0.1.0  
**Deployed to:** gwc.rbee.dev
