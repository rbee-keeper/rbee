# TEAM-423 Step 1: Fix RbeeVsOllamaPage (Critical Blocker)

**Created by:** TEAM-423  
**Date:** 2025-11-08  
**Status:** PENDING  
**Priority:** 🔴 CRITICAL  
**Estimated Time:** 1 hour

---

## 🎯 Objective

Fix component serialization issues in RbeeVsOllamaPage to unblock the build.

**Root Cause:** Component references in props (e.g., `icon: Server` instead of `icon: 'Server'`)

---

## 📋 Tasks

### Task 1.1: Identify Component References (15 min)

**Commands:**
```bash
cd frontend/apps/commercial/components/pages/RbeeVsOllamaPage

# Find component assignments
grep -n ": [A-Z]" RbeeVsOllamaPageProps.tsx

# Find Lucide icon imports
grep "from 'lucide-react'" RbeeVsOllamaPageProps.tsx

# Find CodeBlock usage
grep -B 2 -A 5 "CodeBlock" RbeeVsOllamaPageProps.tsx
```

**What to look for:**
- Lines with `icon: Server` (should be `icon: 'Server'`)
- Lines with `component: SomeComponent` (should be config or JSX)
- Any uppercase identifiers assigned to props without `<>`

### Task 1.2: Fix Icon References (15 min)

**Pattern:**
```typescript
// ❌ WRONG
icon: Server
icon: AlertTriangle

// ✅ CORRECT
icon: 'Server'
icon: 'AlertTriangle'
```

**Action:** Manually edit RbeeVsOllamaPageProps.tsx and convert all component references to strings.

### Task 1.3: Fix Component Props (15 min)

If any props reference components directly:

```typescript
// ❌ WRONG
visual: CodeBlock

// ✅ CORRECT - Option A: Render it
visual: <CodeBlock code="..." language="typescript" />

// ✅ BETTER - Option B: Config object
visualConfig: {
  type: 'code',
  code: '...',
  language: 'typescript'
}
```

### Task 1.4: Test the Fix (15 min)

```bash
cd frontend/apps/commercial

# TypeScript check
pnpm run typecheck | grep RbeeVsOllama

# Build test
pnpm build 2>&1 | tee /tmp/build.log

# Verify fix
grep "rbee-vs-ollama" /tmp/build.log
```

---

## ✅ Success Criteria

- [ ] No "rbee-vs-ollama" in build errors
- [ ] Build progresses past this page (may fail on different page)
- [ ] TypeScript compiles without RbeeVsOllama errors
- [ ] All icon references are strings
- [ ] No component references in props

---

## 📝 Implementation Notes

**TEAM-423 signature:** Add to all modified files:
```typescript
// TEAM-423: Fixed component serialization for SSG
```

**Files to modify:**
- `frontend/apps/commercial/components/pages/RbeeVsOllamaPage/RbeeVsOllamaPageProps.tsx`

---

## 🔄 Handoff to Step 2

Once complete:
- Update TEAM_423_PROGRESS.md
- Mark Step 1 as complete
- Proceed to Step 2 (other comparison pages)

---

**Status:** READY TO IMPLEMENT  
**Next:** Execute tasks 1.1-1.4
