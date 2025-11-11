# Phase 5: Testing & Validation

**Status:** 📋 PENDING  
**Dependencies:** Phase 4 (Dev Mode)  
**Estimated Time:** 2-3 hours

---

## Objectives

1. Test manifest generation
2. Validate manifest accuracy
3. Test filter pages
4. Performance benchmarks
5. End-to-end testing

---

## Test Plan

### 1. Manifest Generation Tests

#### Test 1.1: Generate All Manifests

```bash
# Clean slate
rm -rf public/manifests

# Generate manifests
NODE_ENV=production pnpm run generate:manifests

# Verify output
ls -la public/manifests/
```

**Expected:**
- [ ] 13 CivitAI manifests generated
- [ ] 9 HuggingFace manifests generated
- [ ] 1 combined `all-models.json` generated
- [ ] Total: 23 JSON files

#### Test 1.2: Validate Manifest Structure

```bash
# Check combined manifest
cat public/manifests/all-models.json | jq '.'

# Verify structure
cat public/manifests/all-models.json | jq '{
  totalModels,
  civitai,
  huggingface,
  modelCount: (.models | length)
}'
```

**Expected:**
```json
{
  "totalModels": 300,
  "civitai": 150,
  "huggingface": 150,
  "modelCount": 300
}
```

#### Test 1.3: Check for Duplicates

```bash
# Extract all model IDs
cat public/manifests/all-models.json | jq -r '.models[].id' | sort > ids.txt

# Check for duplicates
cat ids.txt | uniq -d
```

**Expected:**
- [ ] No duplicate IDs
- [ ] All IDs unique

#### Test 1.4: Validate Individual Manifests

```bash
# Check CivitAI manifest
cat public/manifests/civitai-filter-checkpoints.json | jq '{
  filter,
  modelCount: (.models | length),
  timestamp
}'

# Check HuggingFace manifest
cat public/manifests/hf-likes.json | jq '{
  filter,
  modelCount: (.models | length),
  timestamp
}'
```

**Expected:**
- [ ] Each manifest has correct filter name
- [ ] Each manifest has 100 models (or less)
- [ ] Each manifest has timestamp

---

### 2. Build Tests

#### Test 2.1: Production Build

```bash
# Clean build
rm -rf .next out

# Production build
NODE_ENV=production pnpm run build

# Check output
ls -la out/models/civitai/
ls -la out/models/huggingface/
```

**Expected:**
- [ ] Build completes without errors
- [ ] ~300 model pages generated
- [ ] No filter combination pages generated
- [ ] Build time < 5 minutes

#### Test 2.2: Dev Build

```bash
# Dev build
NODE_ENV=development pnpm run build:dev
```

**Expected:**
- [ ] Build completes without errors
- [ ] ~20 model pages generated
- [ ] No manifests generated
- [ ] Build time < 2 minutes

#### Test 2.3: Page Count Verification

```bash
# Count generated pages
find out/models/civitai -name "*.html" | wc -l
find out/models/huggingface -name "*.html" | wc -l

# Total pages
find out -name "*.html" | wc -l
```

**Expected:**
- [ ] CivitAI pages: ~150
- [ ] HuggingFace pages: ~150
- [ ] Total pages: ~300-350 (including index, workers, etc.)

---

### 3. Runtime Tests

#### Test 3.1: Model Detail Pages

```bash
# Start production server
pnpm run start

# Test CivitAI model page
curl http://localhost:3000/models/civitai/civitai-4201

# Test HuggingFace model page
curl http://localhost:3000/models/huggingface/sentence-transformers--all-minilm-l6-v2
```

**Expected:**
- [ ] Pages load successfully
- [ ] No 404 errors
- [ ] Content renders correctly

#### Test 3.2: Filter Pages (Client-Side)

```bash
# Test CivitAI filter page
curl http://localhost:3000/models/civitai/filter/checkpoints

# Test HuggingFace filter page
curl http://localhost:3000/models/huggingface/likes
```

**Expected:**
- [ ] Pages load successfully
- [ ] Manifest JSON is fetched
- [ ] Model grid renders

#### Test 3.3: Manifest Loading

```bash
# Test manifest endpoint
curl http://localhost:3000/manifests/civitai-filter-checkpoints.json
curl http://localhost:3000/manifests/hf-likes.json
curl http://localhost:3000/manifests/all-models.json
```

**Expected:**
- [ ] Manifests are accessible
- [ ] JSON is valid
- [ ] Correct CORS headers

---

### 4. Performance Tests

#### Test 4.1: Build Time Comparison

**Before (with filter prerendering):**
```bash
time pnpm run build
# Expected: ~10 minutes
```

**After (manifest-based):**
```bash
time pnpm run build
# Expected: <5 minutes
```

**Improvement:** >50% faster ✅

#### Test 4.2: Page Load Time

```bash
# Install lighthouse
npm install -g lighthouse

# Test model detail page
lighthouse http://localhost:3000/models/civitai/civitai-4201 --only-categories=performance

# Test filter page
lighthouse http://localhost:3000/models/civitai/filter/checkpoints --only-categories=performance
```

**Expected:**
- [ ] Model detail page: >90 performance score
- [ ] Filter page: >85 performance score
- [ ] First Contentful Paint: <1.5s
- [ ] Time to Interactive: <3s

#### Test 4.3: Manifest Size

```bash
# Check manifest sizes
du -h public/manifests/*.json

# Total size
du -sh public/manifests/
```

**Expected:**
- [ ] Individual manifests: <50KB each
- [ ] Combined manifest: <200KB
- [ ] Total: <2MB

---

### 5. End-to-End Tests

#### Test 5.1: User Journey - Browse Models

1. Visit homepage
2. Click "Browse Models"
3. Select "CivitAI Checkpoints"
4. Click on a model
5. View model details

**Expected:**
- [ ] All pages load correctly
- [ ] No broken links
- [ ] Images load
- [ ] Data is accurate

#### Test 5.2: User Journey - Filter Models

1. Visit `/models/civitai`
2. Apply filter: "SDXL"
3. Apply sort: "Most Downloaded"
4. Click on a model

**Expected:**
- [ ] Filter page loads manifest
- [ ] Models display correctly
- [ ] Sorting works
- [ ] Model detail page loads

#### Test 5.3: User Journey - Search

1. Visit `/search`
2. Search for "stable diffusion"
3. Click on a result

**Expected:**
- [ ] Search works
- [ ] Results are relevant
- [ ] Model pages load

---

### 6. Error Handling Tests

#### Test 6.1: Missing Manifest

```bash
# Remove a manifest
rm public/manifests/civitai-filter-checkpoints.json

# Visit filter page
curl http://localhost:3000/models/civitai/filter/checkpoints
```

**Expected:**
- [ ] Page doesn't crash
- [ ] Fallback to live API
- [ ] Models still display

#### Test 6.2: Invalid Manifest

```bash
# Corrupt a manifest
echo "invalid json" > public/manifests/hf-likes.json

# Visit filter page
curl http://localhost:3000/models/huggingface/likes
```

**Expected:**
- [ ] Page doesn't crash
- [ ] Error is logged
- [ ] Fallback to live API

#### Test 6.3: API Failure

```bash
# Block API access (simulate network failure)
# Visit filter page in dev mode
```

**Expected:**
- [ ] Error message displays
- [ ] Retry button works
- [ ] User is informed

---

## Automated Tests

### Unit Tests

**File:** `lib/__tests__/manifests.test.ts`

```typescript
import { describe, it, expect } from 'vitest'
import { loadAllModels, loadModelsBySource } from '../manifests'

describe('Manifest Loader', () => {
  it('loads all models', async () => {
    const models = await loadAllModels()
    expect(models).toBeDefined()
    expect(models.length).toBeGreaterThan(0)
  })
  
  it('filters by source', async () => {
    const civitaiModels = await loadModelsBySource('civitai')
    expect(civitaiModels.every(m => m.source === 'civitai')).toBe(true)
  })
  
  it('has no duplicate IDs', async () => {
    const models = await loadAllModels()
    const ids = models.map(m => m.id)
    const uniqueIds = new Set(ids)
    expect(ids.length).toBe(uniqueIds.size)
  })
})
```

### Integration Tests

**File:** `__tests__/filter-pages.test.tsx`

```typescript
import { describe, it, expect } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import CivitAIFilterPage from '@/app/models/civitai/[...filter]/page'

describe('Filter Pages', () => {
  it('loads and displays models', async () => {
    render(<CivitAIFilterPage params={{ filter: ['filter', 'checkpoints'] }} />)
    
    await waitFor(() => {
      expect(screen.getByText(/Loading/)).toBeInTheDocument()
    })
    
    await waitFor(() => {
      expect(screen.queryByText(/Loading/)).not.toBeInTheDocument()
    })
    
    expect(screen.getByText(/CivitAI Models/)).toBeInTheDocument()
  })
})
```

---

## Test Checklist

### Manifest Generation
- [ ] All manifests generated
- [ ] No duplicates
- [ ] Correct structure
- [ ] Valid JSON

### Build Process
- [ ] Production build works
- [ ] Dev build works
- [ ] Correct page count
- [ ] Build time improved

### Runtime
- [ ] Model pages load
- [ ] Filter pages load
- [ ] Manifests accessible
- [ ] No errors

### Performance
- [ ] Build time <5min
- [ ] Page load <3s
- [ ] Manifest size <2MB
- [ ] Lighthouse score >85

### Error Handling
- [ ] Missing manifest handled
- [ ] Invalid manifest handled
- [ ] API failure handled
- [ ] User feedback works

---

## Success Criteria

✅ Phase 5 is complete when:
1. All tests pass
2. No regressions found
3. Performance improved
4. Error handling works
5. Documentation updated

---

## Next Phase

Once Phase 5 is complete, proceed to:
**[Phase 6: Deployment](./PHASE_6_DEPLOYMENT.md)**
