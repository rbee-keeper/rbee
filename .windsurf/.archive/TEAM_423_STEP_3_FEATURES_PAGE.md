# TEAM-423 Step 3: Fix FeaturesPage

**Created by:** TEAM-423  
**Date:** 2025-11-08  
**Status:** PENDING  
**Priority:** 🟠 HIGH  
**Estimated Time:** 1.5 hours

---

## 🎯 Objective

Fix 20 JSX props in FeaturesPage by converting them to serializable formats.

**Root Cause:** JSX content in props (e.g., `content: (<div>...</div>)`)

---

## 📋 Tasks

### Task 3.1: Analyze JSX Props (20 min)

**Commands:**
```bash
cd frontend/apps/commercial/components/pages/FeaturesPage

# Find all JSX props
grep -n ": (" FeaturesPageProps.tsx | head -20

# Count JSX props
grep -c ": (" FeaturesPageProps.tsx

# Find content props
grep -B 2 -A 5 "content: (" FeaturesPageProps.tsx
```

**Document findings:**
- How many `content` props?
- How many `decoration` props?
- How many other JSX props?

### Task 3.2: Convert Content Props (40 min)

**Pattern:**
```typescript
// ❌ WRONG
content: (
  <div className="prose">
    <p>Feature description</p>
    <ul>
      <li>Point 1</li>
      <li>Point 2</li>
    </ul>
  </div>
)

// ✅ CORRECT - Option A: Plain string
content: "Feature description\n\n- Point 1\n- Point 2"

// ✅ BETTER - Option B: Config object
contentConfig: {
  type: 'prose',
  text: 'Feature description',
  points: ['Point 1', 'Point 2']
}
```

**Action:** Convert all content props to strings or config objects.

### Task 3.3: Convert Decoration Props (20 min)

**Pattern:**
```typescript
// ❌ WRONG
decoration: (
  <div className="absolute">
    <SomeComponent />
  </div>
)

// ✅ CORRECT - Config object
decorationConfig: {
  type: 'component',
  component: 'SomeComponent',
  position: 'absolute'
}
```

### Task 3.4: Update Page Component (20 min)

If using config objects, update the page component to render from config:

```typescript
// In FeaturesPage.tsx
import { renderContent } from '../../utils/contentRenderer'

<Template 
  {...props} 
  content={renderContent(props.contentConfig)}
/>
```

**Note:** Only if needed. Prefer plain strings if possible.

### Task 3.5: Test the Fix (10 min)

```bash
cd frontend/apps/commercial

# TypeScript check
pnpm run typecheck | grep FeaturesPage

# Build test
pnpm build 2>&1 | tee /tmp/build.log

# Verify fix
grep "features" /tmp/build.log
```

---

## ✅ Success Criteria

- [ ] All 20 JSX props converted
- [ ] No `content: (` patterns in Props file
- [ ] No `decoration: (` patterns in Props file
- [ ] TypeScript compiles
- [ ] Build progresses past FeaturesPage
- [ ] Page renders correctly (verify after build)

---

## 📝 Implementation Notes

**TEAM-423 signature:** Add to all modified files:
```typescript
// TEAM-423: Converted JSX props to serializable format for SSG
```

**Files to modify:**
- `frontend/apps/commercial/components/pages/FeaturesPage/FeaturesPageProps.tsx`
- `frontend/apps/commercial/components/pages/FeaturesPage/FeaturesPage.tsx` (if config objects used)

**Pattern preference:**
1. Plain strings (simplest)
2. Config objects (if complex structure needed)
3. Avoid creating new components unless absolutely necessary

---

## 🔄 Handoff to Step 4

Once complete:
- Update TEAM_423_PROGRESS.md
- Mark Step 3 as complete
- Proceed to Step 4 (FAQ pages)

---

**Status:** READY TO IMPLEMENT  
**Prerequisite:** Steps 1-2 complete  
**Next:** Execute tasks 3.1-3.5
