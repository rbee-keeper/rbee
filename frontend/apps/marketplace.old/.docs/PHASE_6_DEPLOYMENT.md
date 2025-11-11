# Phase 6: Deployment

**Status:** 📋 PENDING  
**Dependencies:** Phase 5 (Testing)  
**Estimated Time:** 1-2 hours

---

## Objectives

1. Update CI/CD pipeline
2. Deploy to staging
3. Verify production build
4. Deploy to production
5. Monitor and validate

---

## Pre-Deployment Checklist

### Code Review
- [ ] All phases completed
- [ ] All tests passing
- [ ] No console errors
- [ ] Code reviewed
- [ ] Documentation updated

### Performance Validation
- [ ] Build time <5 minutes
- [ ] Page count reduced (~300 pages)
- [ ] Manifest size <2MB
- [ ] Lighthouse score >85

### Functionality Validation
- [ ] Model detail pages work
- [ ] Filter pages work
- [ ] Manifests load correctly
- [ ] Error handling works
- [ ] Dev mode works

---

## Deployment Steps

### Step 1: Update CI/CD Pipeline

**File:** `.github/workflows/deploy-marketplace.yml` (or similar)

```yaml
name: Deploy Marketplace

on:
  push:
    branches:
      - main
      - production

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
      
      - name: Install pnpm
        uses: pnpm/action-setup@v2
        with:
          version: 8
      
      - name: Install dependencies
        run: pnpm install --frozen-lockfile
      
      - name: Generate manifests
        run: |
          cd frontend/apps/marketplace
          NODE_ENV=production pnpm run generate:manifests
        env:
          NODE_ENV: production
      
      - name: Build marketplace
        run: |
          cd frontend/apps/marketplace
          pnpm run build
        env:
          NODE_ENV: production
          NEXT_PUBLIC_SITE_URL: https://marketplace.rbee.dev
      
      - name: Deploy to Cloudflare Pages
        uses: cloudflare/pages-action@v1
        with:
          apiToken: ${{ secrets.CLOUDFLARE_API_TOKEN }}
          accountId: ${{ secrets.CLOUDFLARE_ACCOUNT_ID }}
          projectName: rbee-marketplace
          directory: frontend/apps/marketplace/out
          branch: ${{ github.ref_name }}
```

### Step 2: Deploy to Staging

```bash
# Create staging branch
git checkout -b staging

# Deploy using xtask
cargo xtask release --app marketplace --type patch --ci

# Or manually
cd frontend/apps/marketplace
NODE_ENV=production pnpm run build
npx wrangler pages deploy out/ --project-name=rbee-marketplace --branch=staging
```

**Staging URL:** `https://staging.rbee-marketplace.pages.dev`

### Step 3: Staging Validation

#### 3.1: Verify Manifests

```bash
# Check manifests are deployed
curl https://staging.rbee-marketplace.pages.dev/manifests/all-models.json | jq '.totalModels'

# Check individual manifests
curl https://staging.rbee-marketplace.pages.dev/manifests/civitai-filter-checkpoints.json | jq '.models | length'
```

**Expected:**
- [ ] All manifests accessible
- [ ] Correct model counts
- [ ] Valid JSON

#### 3.2: Test Model Pages

```bash
# Test CivitAI model
curl -I https://staging.rbee-marketplace.pages.dev/models/civitai/civitai-4201

# Test HuggingFace model
curl -I https://staging.rbee-marketplace.pages.dev/models/huggingface/sentence-transformers--all-minilm-l6-v2
```

**Expected:**
- [ ] 200 OK responses
- [ ] Pages load correctly
- [ ] No 404 errors

#### 3.3: Test Filter Pages

Visit in browser:
- `https://staging.rbee-marketplace.pages.dev/models/civitai/filter/checkpoints`
- `https://staging.rbee-marketplace.pages.dev/models/huggingface/likes`

**Expected:**
- [ ] Pages load
- [ ] Manifests fetch correctly
- [ ] Model grids render
- [ ] No console errors

#### 3.4: Performance Test

```bash
# Run Lighthouse
lighthouse https://staging.rbee-marketplace.pages.dev/models/civitai/civitai-4201 --only-categories=performance

# Check page load time
curl -w "@curl-format.txt" -o /dev/null -s https://staging.rbee-marketplace.pages.dev/models/civitai/filter/checkpoints
```

**Expected:**
- [ ] Performance score >85
- [ ] Page load <3s
- [ ] No performance regressions

---

### Step 4: Production Deployment

#### 4.1: Final Checks

```bash
# Run all tests
pnpm run test

# Type check
pnpm run type-check

# Lint
pnpm run lint

# Build locally
NODE_ENV=production pnpm run build
```

**Expected:**
- [ ] All tests pass
- [ ] No type errors
- [ ] No lint errors
- [ ] Build succeeds

#### 4.2: Deploy to Production

```bash
# Merge to production branch
git checkout production
git merge staging

# Deploy
cargo xtask release --app marketplace --type patch --ci

# Or manually
cd frontend/apps/marketplace
NODE_ENV=production pnpm run build
npx wrangler pages deploy out/ --project-name=rbee-marketplace --branch=production
```

**Production URL:** `https://marketplace.rbee.dev`

#### 4.3: Verify Production

```bash
# Check manifests
curl https://marketplace.rbee.dev/manifests/all-models.json | jq '.totalModels'

# Test model pages
curl -I https://marketplace.rbee.dev/models/civitai/civitai-4201
curl -I https://marketplace.rbee.dev/models/huggingface/sentence-transformers--all-minilm-l6-v2

# Test filter pages
curl -I https://marketplace.rbee.dev/models/civitai/filter/checkpoints
curl -I https://marketplace.rbee.dev/models/huggingface/likes
```

**Expected:**
- [ ] All endpoints return 200 OK
- [ ] Manifests load correctly
- [ ] Pages render correctly

---

## Post-Deployment Monitoring

### 1. Cloudflare Analytics

Monitor:
- [ ] Page views
- [ ] Error rate
- [ ] Response times
- [ ] Cache hit ratio

**Expected:**
- Error rate: <1%
- Cache hit ratio: >90%
- Response time: <500ms

### 2. Error Tracking

Check for:
- [ ] 404 errors (broken links)
- [ ] 500 errors (server errors)
- [ ] Client-side errors (console errors)

**Action:** Fix any errors immediately

### 3. User Feedback

Monitor:
- [ ] User reports
- [ ] Social media mentions
- [ ] Support tickets

**Action:** Address feedback quickly

---

## Rollback Plan

### If Issues Found

```bash
# Revert to previous deployment
npx wrangler pages deployment list --project-name=rbee-marketplace

# Rollback to specific deployment
npx wrangler pages deployment rollback <DEPLOYMENT_ID> --project-name=rbee-marketplace
```

### If Critical Bug

```bash
# Revert git changes
git revert <commit-hash>

# Redeploy previous version
cargo xtask release --app marketplace --type patch --ci
```

---

## Success Metrics

### Before Deployment
- ~500 pages prerendered
- ~10 minute build times
- Broken filter routes

### After Deployment
- ~300 unique model pages prerendered
- <5 minute build times
- All filter routes work
- Faster page loads

### Validation

```bash
# Check page count
find out -name "*.html" | wc -l
# Expected: ~300-350

# Check build time
time pnpm run build
# Expected: <5 minutes

# Check manifest size
du -sh public/manifests/
# Expected: <2MB
```

---

## Documentation Updates

### Update README

**File:** `frontend/apps/marketplace/README.md`

```markdown
## Build Process

### Development
```bash
pnpm run dev  # Fast, no manifest generation
```

### Production
```bash
pnpm run build  # Generates manifests, builds site
```

### Manifest Generation

Manifests are generated at build time:
- CivitAI: 13 filter combinations
- HuggingFace: 9 filter combinations
- Combined: ~300 unique models

Manifests are cached and served from CDN.
```

### Update CHANGELOG

**File:** `CHANGELOG.md`

```markdown
## [0.1.13] - 2025-11-10

### Changed
- Replaced filter page prerendering with manifest-based system
- Reduced build time from 10min to <5min
- Reduced page count from ~500 to ~300
- Improved filter page performance (client-side manifest loading)

### Added
- JSON manifests for all filter combinations
- Client-side manifest loading
- Dev mode optimization (skips manifest generation)
- Fallback to live API if manifest missing

### Fixed
- Broken filter combination routes
- Slow build times
- Duplicate model pages
```

---

## Communication Plan

### Internal Team

**Slack/Discord Message:**
```
🚀 Marketplace Deployment - Manifest-Based SSG

We've deployed a major improvement to the marketplace build system:

✅ Build time: 10min → <5min (50% faster)
✅ Page count: ~500 → ~300 (40% reduction)
✅ Filter pages: Now client-side (faster loads)
✅ All filter routes: Fixed and working

Technical details:
- Manifests generated at build time
- Only unique model pages prerendered
- Filter pages load manifests client-side
- Dev mode optimized (no manifest generation)

Please test and report any issues!
```

### Users (if applicable)

**Blog Post / Announcement:**
```
We've improved the marketplace performance:

- Faster page loads
- More reliable filter pages
- Better search experience

All your favorite models are still here, just faster! 🚀
```

---

## Success Criteria

✅ Phase 6 is complete when:
1. Staging deployment successful
2. Production deployment successful
3. All validation checks pass
4. No critical errors
5. Performance metrics met
6. Documentation updated
7. Team notified

---

## Final Checklist

### Pre-Deployment
- [ ] All tests pass
- [ ] Code reviewed
- [ ] Documentation updated
- [ ] Staging tested

### Deployment
- [ ] CI/CD updated
- [ ] Staging deployed
- [ ] Staging validated
- [ ] Production deployed
- [ ] Production validated

### Post-Deployment
- [ ] Monitoring active
- [ ] No errors detected
- [ ] Performance metrics met
- [ ] Team notified
- [ ] Users informed (if needed)

---

## 🎉 Project Complete!

Once all checklist items are complete, the Manifest-Based SSG system is fully deployed and operational!

**Return to:** [MANIFEST_BASED_SSG_MASTERPLAN.md](./MANIFEST_BASED_SSG_MASTERPLAN.md)
