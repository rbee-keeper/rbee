# SSG Step 5: Remove force-dynamic and Restore SSG

**Step:** 5 of 5  
**Time:** 1 hour  
**Priority:** 🔴 CRITICAL  
**Status:** PENDING (requires Step 4 completion)

---

## 🎯 Goal

Remove all `force-dynamic` declarations and verify SSG works correctly.

---

## 📊 Scope

**Files with force-dynamic:** 24 pages

**Commercial Frontend:**
1. `/app/page.tsx`
2. `/app/pricing/page.tsx`
3. `/app/features/page.tsx`
4. `/app/legal/page.tsx`
5. `/app/legal/privacy/page.tsx`
6. `/app/legal/terms/page.tsx`
7. `/app/compare/page.tsx`
8. `/app/compare/rbee-vs-ollama/page.tsx`
9. `/app/compare/rbee-vs-vllm/page.tsx`
10. `/app/compare/rbee-vs-together-ai/page.tsx`
11. `/app/compare/rbee-vs-ray-kserve/page.tsx`
12. `/app/features/multi-machine/page.tsx`
13. `/app/features/openai-compatible/page.tsx`
14. `/app/features/rhai-scripting/page.tsx`
15. `/app/features/gdpr-compliance/page.tsx`
16. `/app/features/ssh-deployment/page.tsx`
17. `/app/features/heterogeneous-hardware/page.tsx`
18. `/app/use-cases/page.tsx`
19. `/app/use-cases/homelab/page.tsx`
20. `/app/use-cases/academic/page.tsx`
21. `/app/gpu-providers/page.tsx`
22. `/app/earn/page.tsx`
23. `/app/debug-env/page.tsx`
24. `/app/not-found.tsx`

**Marketplace Frontend:**
1. `/app/models/huggingface/[...filter]/page.tsx`

---

## 📋 Tasks

### Task 5.1: Verify All JSX is Fixed (15 min)

**CRITICAL: Do not proceed unless Steps 1-4 are complete!**

**Verification checklist:**
```bash
cd /home/vince/Projects/llama-orch/frontend/apps/commercial/components/pages

# Should return 0
grep -r ": (" . --include="*Props.tsx" | wc -l

# Should return 0 (or only type imports)
grep -r "^import { [A-Z]" . --include="*Props.tsx" | grep -v "^import type" | wc -l
```

**If any JSX remains:**
- ❌ STOP - Go back to Steps 2-4
- ❌ Do NOT proceed to remove force-dynamic
- ❌ Build will fail if you proceed

### Task 5.2: Remove force-dynamic (Phase 1 - Critical Pages) (15 min)

**Start with critical pages first:**

```bash
cd /home/vince/Projects/llama-orch/frontend/apps/commercial/app

# Remove from homepage
sed -i '/TEAM-423.*Disable SSG/d' page.tsx
sed -i '/export const dynamic/d' page.tsx

# Remove from pricing
sed -i '/TEAM-423.*Disable SSG/d' pricing/page.tsx
sed -i '/export const dynamic/d' pricing/page.tsx

# Remove from features
sed -i '/TEAM-423.*Disable SSG/d' features/page.tsx
sed -i '/export const dynamic/d' features/page.tsx
```

**Test immediately:**
```bash
cd /home/vince/Projects/llama-orch/frontend/apps/commercial
pnpm build 2>&1 | tee /tmp/ssg-test-phase1.log
```

**If build fails:**
- Check error message
- Identify which page failed
- Check that page's props file for remaining JSX
- Fix the JSX
- Try again

### Task 5.3: Remove force-dynamic (Phase 2 - High Priority) (15 min)

**If Phase 1 succeeds, continue:**

```bash
cd /home/vince/Projects/llama-orch/frontend/apps/commercial/app

# Legal pages
for file in legal/page.tsx legal/privacy/page.tsx legal/terms/page.tsx; do
  sed -i '/TEAM-423.*Disable SSG/d' "$file"
  sed -i '/export const dynamic/d' "$file"
done

# Comparison pages
for file in compare/page.tsx compare/*/page.tsx; do
  sed -i '/TEAM-423.*Disable SSG/d' "$file"
  sed -i '/export const dynamic/d' "$file"
done
```

**Test:**
```bash
pnpm build 2>&1 | tee /tmp/ssg-test-phase2.log
```

### Task 5.4: Remove force-dynamic (Phase 3 - Medium Priority) (15 min)

**If Phase 2 succeeds, continue:**

```bash
cd /home/vince/Projects/llama-orch/frontend/apps/commercial/app

# Feature detail pages
for file in features/*/page.tsx; do
  sed -i '/TEAM-423.*Disable SSG/d' "$file"
  sed -i '/export const dynamic/d' "$file"
done

# Use case pages
for file in use-cases/page.tsx use-cases/*/page.tsx; do
  sed -i '/TEAM-423.*Disable SSG/d' "$file"
  sed -i '/export const dynamic/d' "$file"
done

# Other pages
for file in gpu-providers/page.tsx earn/page.tsx; do
  sed -i '/TEAM-423.*Disable SSG/d' "$file"
  sed -i '/export const dynamic/d' "$file"
done
```

**Test:**
```bash
pnpm build 2>&1 | tee /tmp/ssg-test-phase3.log
```

### Task 5.5: Remove force-dynamic (Phase 4 - Low Priority) (10 min)

**If Phase 3 succeeds, finish:**

```bash
cd /home/vince/Projects/llama-orch/frontend/apps/commercial/app

# Debug and not-found
sed -i '/TEAM-423.*Disable SSG/d' debug-env/page.tsx
sed -i '/export const dynamic/d' debug-env/page.tsx

sed -i '/TEAM-423.*Disable SSG/d' not-found.tsx
sed -i '/export const dynamic/d' not-found.tsx
```

**Final test:**
```bash
pnpm build 2>&1 | tee /tmp/ssg-test-final.log
```

### Task 5.6: Verify SSG Output (10 min)

**Check that static HTML is generated:**

```bash
cd /home/vince/Projects/llama-orch/frontend/apps/commercial

# Check .next/server/app directory
ls -la .next/server/app/*.html
ls -la .next/server/app/*/*.html

# Should see .html files for all pages
```

**Verify in build output:**
```
Route (app)                              Size     First Load JS
┌ ○ /                                    ...      ...
├ ○ /pricing                             ...      ...
├ ○ /features                            ...      ...
...

○  (Static)  prerendered as static content
```

**Legend:**
- `○` = Static (SSG) ✅
- `ƒ` = Dynamic (SSR) ❌

**All pages should show `○`!**

---

## ✅ Completion Criteria

- [ ] All JSX removed from props files (verified)
- [ ] All `force-dynamic` removed from page files
- [ ] Build completes successfully
- [ ] All pages show `○` (Static) in build output
- [ ] Static HTML files generated in `.next/server/app/`
- [ ] No runtime errors
- [ ] Pages render correctly in browser

---

## 🔍 Troubleshooting

### Build fails after removing force-dynamic

**Error:** "Functions cannot be passed directly to Client Components"

**Cause:** JSX still exists in a props file

**Fix:**
1. Check error message for page name
2. Find that page's props file
3. Search for `: (` patterns
4. Fix the JSX
5. Try build again

### Build succeeds but pages show `ƒ` instead of `○`

**Cause:** Page has `'use client'` or other dynamic features

**Fix:**
1. Check if page really needs to be dynamic
2. If yes, that's okay (some pages can be dynamic)
3. If no, remove `'use client'` directive

### Static HTML not generated

**Cause:** Build succeeded but SSG didn't run

**Fix:**
1. Check Next.js config
2. Verify no global `force-dynamic` in layout
3. Check for errors in build log

---

## 📊 Success Metrics

### Before (Current State)
- **Static pages:** 0
- **Dynamic pages:** 24
- **Build time:** Fast (no SSG work)
- **Page load:** Slower (SSR on every request)

### After (Target State)
- **Static pages:** 24
- **Dynamic pages:** 0
- **Build time:** Slower (SSG work)
- **Page load:** Faster (pre-rendered HTML)

---

## 🎉 Final Verification

**Run full build:**
```bash
cd /home/vince/Projects/llama-orch
sh scripts/build-all.sh
```

**Should see:**
```
✓ Rust built
✓ Frontend built (all 3 apps)
✓ Build complete! 🐝
```

**Check build output for SSG:**
```
Route (app)                              Size     First Load JS
┌ ○ /                                    ...      ...
├ ○ /pricing                             ...      ...
├ ○ /features                            ...      ...
├ ○ /legal                               ...      ...
├ ○ /legal/privacy                       ...      ...
├ ○ /legal/terms                         ...      ...
...
```

**All routes should have `○` symbol!**

---

## 🔄 Handoff

Once SSG is fully restored:
1. Update `SSG_RESTORATION_MASTER_PLAN.md` with COMPLETE status
2. Create `SSG_RESTORATION_COMPLETE.md` summary
3. Delete all `force-dynamic` references
4. Celebrate! 🎉

---

**Status:** PENDING  
**Prerequisite:** Steps 1-4 complete (all JSX removed)  
**Next:** Remove force-dynamic and verify SSG works
