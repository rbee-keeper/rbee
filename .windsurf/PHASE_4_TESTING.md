# Phase 4: End-to-End Testing

**Status**: 🟡 Not Started  
**Estimated Time**: 1 hour  
**Dependencies**: Phase 3 Complete  
**Blocking**: Phase 5

---

## Objectives

1. ✅ Test single filters work (Small/Medium/Large)
2. ✅ Test multiple filters work together
3. ✅ Test filter removal (clicking default values)
4. ✅ Test manifest loading
5. ✅ Test metadata display
6. ✅ Verify no infinite loops
7. ✅ Document any remaining issues

---

## Test Suite

### Test 1: Single Filter - Size

**Steps**:
1. Navigate to `http://localhost:7823/models/huggingface`
2. Click "Model Size" dropdown
3. Click "Small (<7B)"
4. Verify URL changes to `?size=small`
5. Verify page shows different models
6. Verify downloads/likes/author display

**Expected Results**:
```
✅ URL: http://localhost:7823/models/huggingface?size=small
✅ Description: "Most Downloaded · Small Models"
✅ First model: sentence-transformers/all-MiniLM-L6-v2
✅ Model count: ~100 models
✅ Metadata shows: "138.1M downloads", "sentence-transformers"
```

**Puppeteer Test**:
```javascript
await page.goto('http://localhost:7823/models/huggingface')
await page.waitForSelector('tbody tr')

// Click Model Size
await page.evaluate(() => {
  const buttons = Array.from(document.querySelectorAll('button'))
  const sizeBtn = buttons.find(btn => btn.textContent?.includes('Model Size'))
  sizeBtn?.click()
})
await page.waitForTimeout(300)

// Click Small
await page.evaluate(() => {
  const items = Array.from(document.querySelectorAll('[role="menuitem"]'))
  const smallOpt = items.find(item => item.textContent?.includes('Small'))
  smallOpt?.click()
})
await page.waitForTimeout(1500)

const url = page.url()
assert(url.includes('size=small'), 'URL should include size=small')

const firstModel = await page.evaluate(() => {
  return document.querySelector('tbody tr td')?.textContent?.trim()
})
console.log('✅ First model:', firstModel)
```

---

### Test 2: Single Filter - Sort

**Steps**:
1. Navigate to `http://localhost:7823/models/huggingface`
2. Click "Sort By" dropdown
3. Click "Most Likes"
4. Verify URL changes to `?sort=likes`
5. Verify page shows different models (sorted by likes)

**Expected Results**:
```
✅ URL: http://localhost:7823/models/huggingface?sort=likes
✅ Description: "Most Liked"
✅ First model: deepseek-ai/DeepSeek-R1 (or similar high-likes model)
✅ Model count: ~100 models
```

---

### Test 3: Multiple Filters

**Steps**:
1. Navigate to `http://localhost:7823/models/huggingface`
2. Click "Model Size" → "Small"
3. Wait for URL to update
4. Click "Sort By" → "Most Likes"
5. Verify URL has BOTH params

**Expected Results**:
```
✅ URL: http://localhost:7823/models/huggingface?size=small&sort=likes
✅ Description: "Most Liked · Small Models"
✅ Manifest loaded: /manifests/hf-filter-likes-small.json
✅ First model matches likes/small manifest
✅ Both filter buttons show active state
```

**Puppeteer Test**:
```javascript
// After setting size=small
await page.evaluate(() => {
  const buttons = Array.from(document.querySelectorAll('button'))
  const sortBtn = buttons.find(btn => btn.textContent?.includes('Sort By'))
  sortBtn?.click()
})
await page.waitForTimeout(300)

await page.evaluate(() => {
  const items = Array.from(document.querySelectorAll('[role="menuitem"]'))
  const likesOpt = items.find(item => item.textContent?.includes('Most Likes'))
  likesOpt?.click()
})
await page.waitForTimeout(1500)

const url = page.url()
assert(url.includes('size=small'), 'URL should preserve size=small')
assert(url.includes('sort=likes'), 'URL should add sort=likes')
```

---

### Test 4: Filter Removal

**Steps**:
1. Set URL to `?size=small&sort=likes`
2. Click "Model Size" → "All Sizes"
3. Verify `size` param is removed
4. Verify `sort` param is preserved

**Expected Results**:
```
✅ Before: http://localhost:7823/models/huggingface?size=small&sort=likes
✅ After: http://localhost:7823/models/huggingface?sort=likes
✅ Description: "Most Liked" (no "Small Models")
✅ Model count increases (back to all sizes)
```

---

### Test 5: Manifest Loading

**Steps**:
1. Open DevTools Network tab
2. Navigate to `http://localhost:7823/models/huggingface?size=small`
3. Check network requests

**Expected Results**:
```
✅ Request: GET /manifests/hf-filter-small.json
✅ Status: 200 OK
✅ Content-Type: application/json
✅ Response: { filter: "filter/small", models: [...], timestamp: "..." }
✅ No full page reload (check for document request)
```

---

### Test 6: Metadata Display

**Steps**:
1. Navigate to filtered page
2. Inspect model rows

**Expected Results**:
```
✅ Downloads column shows values like "138.1M", not "0"
✅ Likes column shows values like "4.1K", not "0"
✅ Author column shows values like "sentence-transformers", not "—"
✅ Tags display for models
```

**Check in Browser**:
```javascript
// Console
const rows = Array.from(document.querySelectorAll('tbody tr')).slice(0, 5)
rows.forEach((row, i) => {
  const cells = row.querySelectorAll('td')
  console.log(`Model ${i + 1}:`, {
    name: cells[0]?.textContent?.trim(),
    author: cells[1]?.textContent?.trim(),
    downloads: cells[2]?.textContent?.trim(),
    likes: cells[3]?.textContent?.trim()
  })
})
```

---

### Test 7: No Infinite Loops

**Steps**:
1. Clear browser console
2. Navigate to `?size=small`
3. Click "Most Likes"
4. Wait 5 seconds
5. Check console

**Expected Results**:
```
✅ Console shows 2-3 SSG log messages (initial + manifest load)
✅ No hundreds of repeated messages
✅ No "Maximum call stack size exceeded"
✅ No "Failed to fetch RSC payload" errors
✅ Page remains responsive
```

**Console Pattern (Good)**:
```
[SSG] Fetching top 100 HuggingFace models for initial render
[manifests-client] Loaded 100 models for filter/small
[manifests-client] Loaded 99 models for filter/likes/small
```

**Console Pattern (Bad)**:
```
[SSG] Fetching...
[SSG] Fetching...
[SSG] Fetching... (×500)
```

---

### Test 8: All Size Categories Different

**Steps**:
1. Navigate to `?size=small`, note first 3 models
2. Navigate to `?size=medium`, note first 3 models
3. Navigate to `?size=large`, note first 3 models
4. Compare

**Expected Results**:
```
Small:  sentence-transformers/all-MiniLM-L6-v2, timm/mobilenetv3_small_100...
Medium: omni-research/Tarsier2-Recap-7b, Qwen/Qwen2.5-7B-Instruct...
Large:  FacebookAI/roberta-large, facebook/esm2_t33_650M_UR50D...

✅ All three lists are DIFFERENT
✅ No overlap in first 5 models
```

---

### Test 9: Browser Back/Forward

**Steps**:
1. Navigate through filters: default → small → small+likes
2. Click browser back button
3. Click browser forward button

**Expected Results**:
```
✅ Back: Returns to ?size=small (not default)
✅ Forward: Returns to ?size=small&sort=likes
✅ Data loads correctly for each navigation
✅ No full page reload
```

---

### Test 10: Direct URL Navigation

**Steps**:
1. Manually navigate to `http://localhost:7823/models/huggingface?size=medium&sort=likes`
2. Check page loads correctly

**Expected Results**:
```
✅ Manifest loads: /manifests/hf-filter-likes-medium.json
✅ Filter buttons show correct active state
✅ Description: "Most Liked · Medium Models"
✅ Models match the filter
```

---

## Automated Test Script

**File**: `/home/vince/Projects/rbee/frontend/apps/marketplace/tests/filter-integration.test.ts`

```typescript
import { test, expect } from '@playwright/test'

test.describe('HuggingFace Filter Integration', () => {
  test('should update URL when clicking size filter', async ({ page }) => {
    await page.goto('http://localhost:7823/models/huggingface')
    
    // Click Model Size
    await page.click('button:has-text("Model Size")')
    await page.click('[role="menuitem"]:has-text("Small")')
    
    await expect(page).toHaveURL(/size=small/)
  })
  
  test('should load different models for each size', async ({ page }) => {
    // Get small models
    await page.goto('http://localhost:7823/models/huggingface?size=small')
    await page.waitForSelector('tbody tr')
    const smallFirst = await page.locator('tbody tr:first-child td:first-child').textContent()
    
    // Get medium models
    await page.goto('http://localhost:7823/models/huggingface?size=medium')
    await page.waitForSelector('tbody tr')
    const mediumFirst = await page.locator('tbody tr:first-child td:first-child').textContent()
    
    expect(smallFirst).not.toBe(mediumFirst)
  })
  
  test('should preserve filters when adding new one', async ({ page }) => {
    await page.goto('http://localhost:7823/models/huggingface?size=small')
    
    await page.click('button:has-text("Sort By")')
    await page.click('[role="menuitem"]:has-text("Most Likes")')
    
    await expect(page).toHaveURL(/size=small/)
    await expect(page).toHaveURL(/sort=likes/)
  })
})
```

**Run Tests**:
```bash
cd frontend/apps/marketplace
pnpm test:e2e
```

---

## Completion Checklist

- [ ] Test 1: Single filter (size) ✅
- [ ] Test 2: Single filter (sort) ✅
- [ ] Test 3: Multiple filters ✅
- [ ] Test 4: Filter removal ✅
- [ ] Test 5: Manifest loading ✅
- [ ] Test 6: Metadata display ✅
- [ ] Test 7: No infinite loops ✅
- [ ] Test 8: Size categories different ✅
- [ ] Test 9: Browser back/forward ✅
- [ ] Test 10: Direct URL navigation ✅
- [ ] All Playwright tests pass ✅
- [ ] No console errors ✅
- [ ] Performance acceptable (<1s filter change) ✅

---

## Known Issues Log

Document any issues found during testing:

```markdown
### Issue 1: [Title]
**Severity**: High/Medium/Low
**Description**: ...
**Steps to Reproduce**: ...
**Workaround**: ...
**Status**: Open/Fixed
```

---

## Performance Benchmarks

| Action | Time | Status |
|--------|------|--------|
| Initial page load | <2s | ✅ |
| Filter click → URL update | <100ms | ✅ |
| Manifest fetch | <500ms | ✅ |
| Data enrichment | <100ms | ✅ |
| Total filter change | <1s | ✅ |

---

## Next Phase

Once all checkboxes are complete and no critical issues found, move to **Phase 5: Tauri Integration**.

**Status**: 🟡 → ✅ (update when complete)
