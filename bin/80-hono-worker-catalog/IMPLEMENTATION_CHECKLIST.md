# Worker Catalog Implementation Checklist

**Project:** Hybrid Worker Catalog (Git + Binary Registry)  
**Timeline:** 4 weeks  
**Status:** 📋 NOT STARTED

---

## 📊 Progress Overview

- [ ] Phase 1: Git Catalog Setup (Week 1)
- [ ] Phase 2: Binary Registry (Week 2)
- [ ] Phase 3: Database & Analytics (Week 3)
- [ ] Phase 4: Premium Support (Week 4)

---

## 🗓️ Phase 1: Git Catalog Setup (Week 1)

### Day 1: Repository Setup

- [ ] **Create Git Repository**
  ```bash
  gh repo create veighnsche/rbee-worker-catalog --public
  git clone git@github.com:veighnsche/rbee-worker-catalog.git
  cd rbee-worker-catalog
  ```

- [ ] **Create Master Branch with Index**
  - [ ] Create `README.md` with project overview
  - [ ] Create `catalog.json` with worker list
  - [ ] Create `.github/workflows/` for CI/CD
  - [ ] Commit and push to master

- [ ] **Create Worker Branches**
  - [ ] `llm-worker-rbee-cpu` branch
  - [ ] `llm-worker-rbee-cuda` branch
  - [ ] `llm-worker-rbee-metal` branch
  - [ ] `sd-worker-rbee-cpu` branch (if exists)

### Day 2: Worker Metadata

- [ ] **Define metadata.json Schema**
  - [ ] Document all required fields
  - [ ] Create JSON schema for validation
  - [ ] Add examples

- [ ] **Create metadata.json for Each Worker**
  - [ ] llm-worker-rbee-cpu
  - [ ] llm-worker-rbee-cuda
  - [ ] llm-worker-rbee-metal
  - [ ] Include: name, description, version, platforms, features

- [ ] **Migrate Existing PKGBUILDs**
  - [ ] Copy from `public/pkgbuilds/` to branches
  - [ ] Verify PKGBUILD syntax
  - [ ] Generate `.SRCINFO` files

### Day 3: Hono Integration

- [ ] **Update Hono to Fetch from GitHub**
  - [ ] Create `src/github.ts` module
  - [ ] Implement `fetchCatalogIndex()` function
  - [ ] Implement `fetchWorkerMetadata()` function
  - [ ] Implement `fetchPKGBUILD()` function

- [ ] **Update Routes**
  - [ ] Modify `GET /workers` to fetch from GitHub
  - [ ] Modify `GET /workers/:id` to fetch metadata
  - [ ] Keep `GET /workers/:id/PKGBUILD` working
  - [ ] Add caching (5 min TTL)

- [ ] **Add Error Handling**
  - [ ] Handle GitHub API rate limits
  - [ ] Handle network errors
  - [ ] Add fallback to cached data

### Day 4: Testing & Documentation

- [ ] **Test Git Catalog**
  - [ ] Test fetching catalog index
  - [ ] Test fetching worker metadata
  - [ ] Test fetching PKGBUILDs
  - [ ] Test with different workers

- [ ] **Update Documentation**
  - [ ] Update main README.md
  - [ ] Document Git workflow for maintainers
  - [ ] Document API endpoints
  - [ ] Add examples

### Day 5: CI/CD Setup

- [ ] **GitHub Actions Workflow**
  - [ ] Create `.github/workflows/validate-pkgbuild.yml`
  - [ ] Validate PKGBUILD syntax on push
  - [ ] Check metadata.json schema
  - [ ] Generate .SRCINFO automatically

- [ ] **Auto-Update Catalog Index**
  - [ ] Script to scan all branches
  - [ ] Generate catalog.json
  - [ ] Commit to master on worker updates

---

## 🗓️ Phase 2: Binary Registry (Week 2)

### Day 1: R2 Setup

- [ ] **Create R2 Bucket**
  ```bash
  wrangler r2 bucket create rbee-workers
  wrangler r2 bucket create rbee-workers-preview
  ```

- [ ] **Configure Bucket Permissions**
  - [ ] Public read for open-source workers
  - [ ] Private for premium workers
  - [ ] Set CORS policies

- [ ] **Add R2 Binding to wrangler.jsonc**
  ```jsonc
  "r2_buckets": [
    {
      "binding": "WORKER_BINARIES",
      "bucket_name": "rbee-workers"
    }
  ]
  ```

### Day 2: Binary Upload

- [ ] **Build Worker Binaries**
  - [ ] Build llm-worker-rbee-cpu for x86_64
  - [ ] Build llm-worker-rbee-cpu for aarch64
  - [ ] Build llm-worker-rbee-cuda for x86_64
  - [ ] Build llm-worker-rbee-metal for aarch64

- [ ] **Package Binaries**
  - [ ] Create tar.gz archives
  - [ ] Include LICENSE, README, metadata.json
  - [ ] Generate SHA256 checksums

- [ ] **Upload to R2**
  ```bash
  wrangler r2 object put rbee-workers/llm-worker-rbee-cpu/0.1.0/linux-x86_64.tar.gz \
    --file binary.tar.gz
  ```

### Day 3: Download API

- [ ] **Create Version Endpoints**
  - [ ] `GET /v1/workers/:id/versions`
  - [ ] Return list of available versions
  - [ ] Include platform info

- [ ] **Create Download Endpoint**
  - [ ] `GET /v1/workers/:id/:version/download`
  - [ ] Query param: `platform=linux-x86_64`
  - [ ] Return R2 public URL
  - [ ] Include checksum

- [ ] **Add Checksum Verification**
  - [ ] Store checksums in metadata
  - [ ] Return in download response
  - [ ] Document verification process

### Day 4: Testing

- [ ] **Test Binary Downloads**
  - [ ] Test each worker variant
  - [ ] Test each platform
  - [ ] Verify checksums
  - [ ] Test download speed

- [ ] **Integration Test with rbee-hive**
  - [ ] Update rbee-hive to use new API
  - [ ] Test installation flow
  - [ ] Verify binary works after install

### Day 5: Documentation

- [ ] **Update API Documentation**
  - [ ] Document v1 endpoints
  - [ ] Add examples
  - [ ] Include curl commands

- [ ] **Create Binary Upload Guide**
  - [ ] Document build process
  - [ ] Document packaging format
  - [ ] Document upload procedure

---

## 🗓️ Phase 3: Database & Analytics (Week 3)

### Day 1: D1 Setup

- [ ] **Create D1 Database**
  ```bash
  wrangler d1 create rbee-catalog
  wrangler d1 create rbee-catalog-preview
  ```

- [ ] **Add D1 Binding to wrangler.jsonc**
  ```jsonc
  "d1_databases": [
    {
      "binding": "DB",
      "database_name": "rbee-catalog",
      "database_id": "xxx"
    }
  ]
  ```

- [ ] **Create Database Schema**
  - [ ] Create `schema.sql` with all tables
  - [ ] Run migrations: `wrangler d1 execute rbee-catalog --file=schema.sql`
  - [ ] Verify tables created

### Day 2: Data Sync

- [ ] **Create Sync Script**
  - [ ] Fetch metadata from Git branches
  - [ ] Parse metadata.json
  - [ ] Insert into D1 database
  - [ ] Handle updates

- [ ] **Initial Data Load**
  - [ ] Sync all workers
  - [ ] Sync all versions
  - [ ] Verify data integrity

- [ ] **Auto-Sync on Git Push**
  - [ ] GitHub webhook endpoint
  - [ ] Trigger sync on branch update
  - [ ] Update catalog index

### Day 3: Analytics

- [ ] **Track Downloads**
  - [ ] Add middleware to download endpoint
  - [ ] Record: worker_id, version, platform, timestamp
  - [ ] Hash IP for privacy
  - [ ] Store in D1

- [ ] **Create Stats Endpoint**
  - [ ] `GET /v1/workers/:id/stats`
  - [ ] Return: total downloads, downloads by version
  - [ ] Return: downloads by platform
  - [ ] Add caching

- [ ] **Create Dashboard Data Endpoint**
  - [ ] `GET /v1/stats/overview`
  - [ ] Top downloaded workers
  - [ ] Recent downloads
  - [ ] Platform distribution

### Day 4: Caching

- [ ] **Add KV Namespace**
  ```bash
  wrangler kv:namespace create CACHE
  wrangler kv:namespace create CACHE --preview
  ```

- [ ] **Implement Caching Layer**
  - [ ] Cache catalog index (5 min TTL)
  - [ ] Cache worker metadata (10 min TTL)
  - [ ] Cache stats (1 hour TTL)
  - [ ] Invalidate on updates

### Day 5: Testing & Optimization

- [ ] **Performance Testing**
  - [ ] Measure API response times
  - [ ] Test with high load
  - [ ] Optimize slow queries

- [ ] **Database Optimization**
  - [ ] Add missing indexes
  - [ ] Optimize query plans
  - [ ] Test with large datasets

---

## 🗓️ Phase 4: Premium Support (Week 4)

### Day 1: License System

- [ ] **Design License Token Format**
  - [ ] Define token structure
  - [ ] Implement signing/verification
  - [ ] Add expiration handling

- [ ] **Create License Table**
  - [ ] Add to schema.sql
  - [ ] Run migration
  - [ ] Create indexes

- [ ] **Implement License Verification**
  - [ ] Create `src/auth.ts` module
  - [ ] `verifyLicense()` function
  - [ ] Check expiration
  - [ ] Check worker access

### Day 2: Authentication Middleware

- [ ] **Create Auth Middleware**
  - [ ] Extract token from Authorization header
  - [ ] Verify token signature
  - [ ] Check license validity
  - [ ] Attach license to context

- [ ] **Protect Premium Endpoints**
  - [ ] Apply middleware to download endpoint
  - [ ] Check if worker requires license
  - [ ] Return 403 if invalid

- [ ] **Add License Verification Endpoint**
  - [ ] `POST /v1/licenses/verify`
  - [ ] Return license details
  - [ ] Return allowed workers

### Day 3: Premium Binary Storage

- [ ] **Create Private R2 Bucket**
  ```bash
  wrangler r2 bucket create rbee-workers-premium
  ```

- [ ] **Upload Premium Binaries**
  - [ ] Build premium worker variants
  - [ ] Package with license info
  - [ ] Upload to private bucket

- [ ] **Implement Presigned URLs**
  - [ ] Generate time-limited URLs
  - [ ] 1 hour expiration
  - [ ] Include in download response

### Day 4: Premium Worker Metadata

- [ ] **Create Premium Worker Branch**
  - [ ] `llm-worker-rbee-premium` branch
  - [ ] Add metadata.json with pricing
  - [ ] Add README with features
  - [ ] NO PKGBUILD (binary only)

- [ ] **Update Catalog**
  - [ ] Add premium worker to catalog.json
  - [ ] Mark as requires_license: true
  - [ ] Add pricing information

### Day 5: Testing & Documentation

- [ ] **Test Premium Flow**
  - [ ] Test license verification
  - [ ] Test download with valid license
  - [ ] Test download with invalid license
  - [ ] Test license expiration

- [ ] **Documentation**
  - [ ] Document license system
  - [ ] Document premium installation
  - [ ] Create pricing page content
  - [ ] Add troubleshooting guide

---

## 🎯 Success Criteria

### Phase 1: Git Catalog
- [ ] All workers have metadata.json
- [ ] Hono fetches from GitHub successfully
- [ ] PKGBUILDs validate in CI
- [ ] Documentation complete

### Phase 2: Binary Registry
- [ ] Binaries uploaded to R2
- [ ] Download API works for all platforms
- [ ] Checksums verify correctly
- [ ] rbee-hive can install from binaries

### Phase 3: Database & Analytics
- [ ] D1 database operational
- [ ] Metadata synced from Git
- [ ] Download tracking works
- [ ] Stats endpoint returns data

### Phase 4: Premium Support
- [ ] License verification works
- [ ] Premium downloads require auth
- [ ] Presigned URLs expire correctly
- [ ] Premium workers listed in catalog

---

## 📋 Verification Commands

### Test Git Catalog
```bash
# Fetch catalog index
curl https://worker-catalog.rbee.workers.dev/workers

# Fetch worker metadata
curl https://worker-catalog.rbee.workers.dev/workers/llm-worker-rbee-cpu

# Fetch PKGBUILD
curl https://worker-catalog.rbee.workers.dev/workers/llm-worker-rbee-cpu/PKGBUILD
```

### Test Binary Registry
```bash
# List versions
curl https://worker-catalog.rbee.workers.dev/v1/workers/llm-worker-rbee-cpu/versions

# Download binary
curl https://worker-catalog.rbee.workers.dev/v1/workers/llm-worker-rbee-cpu/0.1.0/download?platform=linux-x86_64
```

### Test Premium
```bash
# Verify license
curl -X POST https://worker-catalog.rbee.workers.dev/v1/licenses/verify \
  -H "Content-Type: application/json" \
  -d '{"token":"rbee_lic_abc123..."}'

# Download premium worker
curl https://worker-catalog.rbee.workers.dev/v1/workers/llm-worker-rbee-premium/0.1.0/download \
  -H "Authorization: Bearer rbee_lic_abc123..."
```

---

## 🚨 Blockers & Risks

### Potential Issues
- [ ] GitHub API rate limits (use caching)
- [ ] R2 egress costs (should be zero, verify)
- [ ] D1 query limits (optimize queries)
- [ ] License token security (use proper signing)

### Mitigation Strategies
- Implement aggressive caching (KV + Cache API)
- Monitor R2 usage and costs
- Optimize database queries early
- Use industry-standard JWT signing

---

## 📝 Notes

- Keep backward compatibility with current API
- Test each phase thoroughly before moving to next
- Document as you go
- Get feedback from users early

**TEAM-402 - Implementation Checklist Complete!** 🎉
