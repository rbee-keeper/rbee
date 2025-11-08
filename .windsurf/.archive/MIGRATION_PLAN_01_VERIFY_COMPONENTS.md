# Step 1: Verify HeroAsides Components

**Phase:** 0 - Setup  
**Time:** 5 minutes  
**Priority:** CRITICAL

## 🎯 Goal

Verify that the HeroAsides components exist and are ready to use.

## ✅ What to Check

### 1. Component Files Exist

```bash
# Check component files
ls -la frontend/apps/commercial/components/organisms/HeroAsides/

# Should show:
# - HeroAsides.tsx (198 lines)
# - index.ts (3 lines)
```

### 2. Component Exports

```bash
# Verify exports
cat frontend/apps/commercial/components/organisms/HeroAsides/index.ts
```

**Expected output:**
```typescript
// TEAM-XXX: Hero aside components export
export * from './HeroAsides'
```

### 3. Component Structure

```bash
# Check HeroAsides.tsx structure
grep -E "^export (type|function)" frontend/apps/commercial/components/organisms/HeroAsides/HeroAsides.tsx
```

**Expected exports:**
- `export type IconAsideConfig`
- `export type ImageAsideConfig`
- `export type CardAsideConfig`
- `export type StatsAsideConfig`
- `export type AsideConfig`
- `export function IconAside`
- `export function ImageAside`
- `export function CardAside`
- `export function StatsAside`
- `export function renderAside`

## 🔍 Verification Commands

```bash
cd frontend/apps/commercial

# 1. Check files exist
test -f components/organisms/HeroAsides/HeroAsides.tsx && echo "✅ HeroAsides.tsx exists" || echo "❌ Missing"
test -f components/organisms/HeroAsides/index.ts && echo "✅ index.ts exists" || echo "❌ Missing"

# 2. Check component count
grep -c "^export function" components/organisms/HeroAsides/HeroAsides.tsx
# Should output: 5 (IconAside, ImageAside, CardAside, StatsAside, renderAside)

# 3. Check type count
grep -c "^export type" components/organisms/HeroAsides/HeroAsides.tsx
# Should output: 5 (IconAsideConfig, ImageAsideConfig, CardAsideConfig, StatsAsideConfig, AsideConfig)

# 4. Verify TypeScript compilation
pnpm run type-check 2>&1 | grep -i "heroasides"
# Should have no errors
```

## 📋 Component Checklist

- [ ] `HeroAsides.tsx` exists (198 lines)
- [ ] `index.ts` exists (3 lines)
- [ ] 5 type exports present
- [ ] 5 function exports present
- [ ] No TypeScript errors
- [ ] Components use proper imports (@rbee/ui/atoms, @rbee/ui/utils)

## 🔧 If Components Missing

If components don't exist, they need to be created first:

```bash
# Check if they were created
git log --oneline --all --grep="HeroAsides" | head -5

# If missing, check documentation
cat .windsurf/HERO_ASIDES_GUIDE.md
```

## ✅ Success Criteria

All checks pass:
- ✅ Both files exist
- ✅ 5 types exported
- ✅ 5 functions exported
- ✅ No TypeScript errors
- ✅ Proper imports used

## 🚀 Next Step

Once verified, proceed to:
**[STEP_02_AUDIT_PAGES.md](./MIGRATION_PLAN_02_AUDIT_PAGES.md)** - Audit current page state

---

**Status:** Verification step  
**Blocking:** Yes - must pass before proceeding  
**Time:** 5 minutes  
**Difficulty:** Easy
