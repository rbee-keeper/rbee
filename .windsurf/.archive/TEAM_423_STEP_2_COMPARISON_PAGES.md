# TEAM-423 Step 2: Fix Other Comparison Pages

**Created by:** TEAM-423  
**Date:** 2025-11-08  
**Status:** PENDING  
**Priority:** 🟠 HIGH  
**Estimated Time:** 30 minutes

---

## 🎯 Objective

Apply the same fixes from Step 1 to the remaining comparison pages.

**Pages to fix:**
1. RbeeVsVllmPage
2. RbeeVsTogetherAiPage
3. RbeeVsRayKservePage

---

## 📋 Tasks

### Task 2.1: Batch Identify Issues (10 min)

**Command:**
```bash
cd frontend/apps/commercial/components/pages

for page in RbeeVsVllmPage RbeeVsTogetherAiPage RbeeVsRayKservePage; do
  echo "=== Checking $page ==="
  grep -n ": [A-Z]" $page/${page}Props.tsx | head -10
  echo ""
done
```

### Task 2.2: Fix Each Page (15 min)

**For each page:**

```bash
cd components/pages/[PageName]

# 1. Check for issues
grep -n ": [A-Z]" [PageName]Props.tsx

# 2. Fix icon references (component → string)
# Edit file manually: icon: Server → icon: 'Server'

# 3. Quick test
cd ../../..
pnpm run typecheck | grep [PageName]
```

**Pattern to apply:**
```typescript
// ❌ WRONG
icon: Server
icon: AlertTriangle
icon: CheckCircle

// ✅ CORRECT
icon: 'Server'
icon: 'AlertTriangle'
icon: 'CheckCircle'
```

### Task 2.3: Test All Fixes (5 min)

```bash
cd frontend/apps/commercial

# TypeScript check
pnpm run typecheck 2>&1 | grep -E "error TS|Found"

# Build test
pnpm build 2>&1 | tee /tmp/build.log

# Check progress
grep "Error occurred prerendering" /tmp/build.log
```

---

## ✅ Success Criteria

### Per Page
- [ ] RbeeVsVllmPage: No component references
- [ ] RbeeVsTogetherAiPage: No component references
- [ ] RbeeVsRayKservePage: No component references
- [ ] All icons converted to strings
- [ ] TypeScript compiles for all pages

### Overall
- [ ] Build progresses past all comparison pages
- [ ] No serialization errors for these pages

---

## 📝 Implementation Notes

**TEAM-423 signature:** Add to all modified files:
```typescript
// TEAM-423: Fixed component serialization for SSG
```

**Files to modify:**
- `frontend/apps/commercial/components/pages/RbeeVsVllmPage/RbeeVsVllmPageProps.tsx`
- `frontend/apps/commercial/components/pages/RbeeVsTogetherAiPage/RbeeVsTogetherAiPageProps.tsx`
- `frontend/apps/commercial/components/pages/RbeeVsRayKservePage/RbeeVsRayKservePageProps.tsx`

---

## 🔄 Handoff to Step 3

Once complete:
- Update TEAM_423_PROGRESS.md
- Mark Step 2 as complete
- Proceed to Step 3 (FeaturesPage)

---

**Status:** READY TO IMPLEMENT  
**Prerequisite:** Step 1 complete  
**Next:** Execute tasks 2.1-2.3
