# SSG Quick Action Plan - Immediate Steps

**Goal:** Unblock the build and enable SSG  
**Time:** 2-4 hours  
**Priority:** 🔴 CRITICAL

---

## 🎯 Immediate Action: Fix MultiMachinePage

**Why:** This page is blocking the entire build

**Current Error:**
```
Error occurred prerendering page "/features/multi-machine"
Error: Functions cannot be passed directly to Client Components
```

### JSX Issues in MultiMachinePage

1. **Line 27:** `subheadline: (JSX)`
2. **Line 46:** `output: (JSX)` 
3. **Line 202:** `content: (JSX)`

---

## 📋 Step-by-Step Fix

### Option 1: Quick Fix (30 minutes)

**Convert to markdown strings:**

```typescript
// Before
subheadline: (
  <>
    Connect your <strong>gaming PC</strong>, Mac Studio, and homelab servers.
  </>
)

// After
subheadline: "Connect your **gaming PC**, Mac Studio, and homelab servers."
```

**Pros:** Fast, simple  
**Cons:** Limited styling

### Option 2: Config Objects (1 hour)

**Create proper config types:**

```typescript
// Create types
type SubheadlineConfig = {
  text: string
  highlights?: string[]
}

type TerminalOutputConfig = {
  lines: Array<{
    text: string
    color?: string
  }>
}

// Use in Props
subheadlineConfig: {
  text: "Connect your gaming PC, Mac Studio, and homelab servers",
  highlights: ["gaming PC"]
}

outputConfig: {
  lines: [
    { text: "$ rbee-keeper status", color: "muted" },
    { text: "→ 3 machines detected", color: "success" }
  ]
}
```

**Pros:** Type-safe, reusable, flexible  
**Cons:** Requires component updates

---

## 🚀 Recommended Approach

**Hybrid:** Quick fix first, then improve

### Phase A: Unblock Build (30 min)

1. Convert `subheadline` to markdown string
2. Convert `output` to simple string
3. Convert `content` to markdown
4. Test build

### Phase B: Improve (1 hour)

1. Create config types
2. Create renderer components
3. Update to use configs
4. Test thoroughly

---

## 📝 Implementation Checklist

### Immediate (Do Now)
- [ ] Backup MultiMachinePageProps.tsx
- [ ] Identify all JSX props (3 found)
- [ ] Convert to markdown strings
- [ ] Test build
- [ ] Verify page renders correctly

### Next (After Build Works)
- [ ] Create config types
- [ ] Create renderer components
- [ ] Update to use configs
- [ ] Document pattern

---

## 🔍 After MultiMachinePage

### Check for Other Blockers

```bash
cd frontend/apps/commercial
pnpm build 2>&1 | grep "Error occurred prerendering"
```

If build succeeds:
- ✅ Move to Phase 2 (High Priority Pages)

If build fails on another page:
- 🔴 Fix that page next
- 📝 Update plan

---

## 📊 Full Migration Order

### Critical (Blocking Build)
1. **MultiMachinePage** ← START HERE

### High Priority (Most JSX)
2. **FeaturesPage** (20 JSX props)
3. **TermsPage** (18 JSX props) - FAQ answers
4. **PrivacyPage** (18 JSX props) - FAQ answers
5. **PricingPage** (12 JSX props)

### Continue with plan...
See [FULL_SSG_ENABLEMENT_PLAN.md](./FULL_SSG_ENABLEMENT_PLAN.md)

---

## 🎯 Success Metrics

### Build Success
```bash
cd frontend/apps/commercial
pnpm build
# Should complete without errors
```

### Page Renders
```bash
pnpm dev
# Visit http://localhost:3000/features/multi-machine
# Should render correctly
```

### SSG Output
```bash
ls -la .next/server/app/features/multi-machine/
# Should contain page.html
```

---

## ⚠️ Common Issues

### Issue 1: Markdown Not Rendering
**Solution:** Add markdown processor to component

### Issue 2: Styling Lost
**Solution:** Use config objects instead of markdown

### Issue 3: Build Still Fails
**Solution:** Check for other JSX props, use grep to find them

---

## 📚 Reference

### Find JSX Props
```bash
grep -n ": (" MultiMachinePageProps.tsx
```

### Test Build
```bash
cd frontend/apps/commercial
pnpm build
```

### Check TypeScript
```bash
pnpm run typecheck
```

---

## 🚀 Ready to Start?

### Command to Begin

```bash
# 1. Navigate to page
cd frontend/apps/commercial/components/pages/MultiMachinePage

# 2. Backup
cp MultiMachinePageProps.tsx MultiMachinePageProps.tsx.backup

# 3. Open in editor
# Start fixing JSX props!
```

---

**Status:** 📋 READY TO EXECUTE  
**Next:** Fix MultiMachinePage  
**Time:** 30 min - 1 hour  
**Impact:** Unblocks entire build
