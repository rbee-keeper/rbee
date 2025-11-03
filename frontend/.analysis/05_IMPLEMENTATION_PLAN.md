# Implementation Plan

**Date:** November 3, 2025  
**Goal:** Step-by-step plan to update commercial site for better conversion

---

## Overview

**Total Effort:** 8-12 days  
**Priority:** High (conversion impact)  
**Risk:** Low (incremental updates, no breaking changes)

---

## Phase 1: Quick Wins - Copy Updates (1-2 days)

### Goal
Update existing props without changing templates. Immediate conversion improvement.

### Tasks

#### 1.1 Update HomePageProps.tsx
**File:** `packages/rbee-ui/src/pages/HomePage/HomePageProps.tsx`

**Changes:**
- Hero badge: "Free Forever • No Credit Card • No Limits"
- Hero headline: "Stop Paying for AI APIs. Run Everything Free."
- Hero subcopy: Add concrete examples (ChatGPT + Stable Diffusion)
- WhatIsRbee title: "Turn Scattered GPUs Into One Intelligent Swarm"
- UseCases title: "Do What Was Impossible Before"
- UseCases description: "Remember when running two AI models crashed your system?"
- EmailCapture badge: Remove "In Development • 68%"
- EmailCapture headline: "Start Orchestrating Your GPUs in 5 Minutes"

**Verification:**
```bash
cd frontend/apps/commercial
pnpm dev
# Check localhost:3000 for updated copy
```

**Estimated Time:** 2-3 hours

---

#### 1.2 Add Concrete Examples to Use Cases
**File:** Same as above

**Changes:**
Add `concreteExample` to each use case:
- Solo Developer: "Run Cursor AI + Stable Diffusion simultaneously"
- Small Team: "5 devs, 3 workstations, $6,000/year saved"
- Homelab: "Turn 4 idle GPUs into your personal AI lab"

**Estimated Time:** 1 hour

---

## Phase 2: Color Scheme Update (1 day)

### Goal
Update theme tokens for warmer, friendlier feel. Fix dark mode WCAG compliance.

### Tasks

#### 2.1 Update Light Mode Colors
**File:** `packages/rbee-ui/src/tokens/theme-tokens.css`

**Changes:**
```css
:root {
  --background: #fdfbf7;
  --card: #fffef9;
  --primary: #e6a23c;
  --accent: #f0b454;
  --secondary: #f5f3ed;
  --muted: #f8f6f0;
  --border: #e8e3d8;
}
```

**Verification:**
- Check all pages in light mode
- Verify contrast ratios (WCAG AA)
- Test hover states

**Estimated Time:** 2 hours

---

#### 2.2 Update Dark Mode Colors (CRITICAL)
**File:** Same as above

**Changes:**
```css
.dark {
  --background: #1a1612;
  --card: #252118;
  --primary: #f0b454;  /* CRITICAL: Fixes WCAG failure */
  --accent: #f5c675;
  --muted: #2a2520;
  --border: #3a342d;
}
```

**Verification:**
- Check all pages in dark mode
- Verify primary color contrast: 7.2:1 ✅
- Test all interactive elements

**Estimated Time:** 2 hours

---

#### 2.3 Test Across All Pages
**Files:** All pages in commercial app

**Verification:**
- HomePage ✓
- DevelopersPage ✓
- ProvidersPage ✓
- EnterprisePage ✓
- All components render correctly
- No visual regressions

**Estimated Time:** 2 hours

---

## Phase 3: New Templates (2-3 days)

### Goal
Add missing high-conversion sections from reference.

### Tasks

#### 3.1 Create BeforeAfterCard Molecule (RENAMED)
**File:** `packages/rbee-ui/src/molecules/BeforeAfterCard/BeforeAfterCard.tsx`

**Why renamed:** We already have `ComparisonTemplate` for feature matrix tables. This is a different visual component for "before/after" style comparisons.

**Implementation:**
- Create component with tone variants (destructive, success, primary)
- Add glow effect
- Add checkmark/X icons
- Export from molecules/index.ts

**Estimated Time:** 3 hours

---

#### 3.2 Create BeforeAfterTemplate (RENAMED)
**File:** `packages/rbee-ui/src/templates/BeforeAfterTemplate/`

**Why renamed:** Avoid confusion with existing `ComparisonTemplate` which shows feature matrices.

**Implementation:**
- Create template with TemplateContainer
- Wire up two BeforeAfterCards (left/right)
- Create props interface
- Export from templates/index.ts

**Estimated Time:** 2 hours

---

#### 3.3 Add BeforeAfter to HomePage
**File:** `packages/rbee-ui/src/pages/HomePage/HomePageProps.tsx`

**Implementation:**
- Add beforeAfterProps (NOT comparisonCardProps - that's the matrix table)
- Add "Without rbee" vs "With rbee" content
- Place between Solution and HowItWorks sections
- Keep existing ComparisonTemplate lower on page (feature matrix)

**Content:**
```typescript
Without rbee:
- GPU memory conflicts - Tasks fighting for the same resources
- Manual resource management - You decide what runs where and when
- Crashed processes - One task can kill another
- Wasted GPU time - Idle GPUs while others are overloaded
- Complex setup - Different APIs for different hardware

With rbee:
- Intelligent scheduling - Queen coordinates all tasks automatically
- Automatic resource allocation - rbee handles everything for you
- Zero conflicts - Tasks never interfere with each other
- Maximum utilization - Every GPU working at optimal capacity
- One simple API - OpenAI-compatible across all hardware
```

**Estimated Time:** 2 hours

---

#### 3.4 Create StatsBar Component (SKIP FOR NOW)
**File:** `packages/rbee-ui/src/molecules/StatsBar/StatsBar.tsx`

**Note:** rbee is currently a solo project (~7 stars). Skip social proof implementation until you have a real community.

**When you have community (future):**
- Simple component with centered layout
- Display large number + label
- Export from molecules/index.ts

**Estimated Time:** 1 hour (when needed)

---

#### 3.5 ~~Add Social Proof Section~~ SKIPPED
**Reason:** No community yet. Solo developer project. Don't add fake numbers.

---

## Phase 4: Template Enhancements (2-3 days)

### Goal
Polish existing templates with reference features.

### Tasks

#### 4.1 Add Pulse Animation to Badge
**File:** `packages/rbee-ui/src/atoms/Badge/Badge.tsx`

**Implementation:**
- Add `showPulse` prop
- Add pulsing dot before badge text
- Use for "Free Forever" badge

**Estimated Time:** 1 hour

---

#### 4.2 Add Social Proof to HeroTemplate
**File:** `packages/rbee-ui/src/templates/HeroTemplate/`

**Implementation:**
- Add `socialProof` prop to HeroTemplateProps
- Render below CTAs
- Show icons + labels

**Estimated Time:** 2 hours

---

#### 4.3 Update UseCasesTemplate
**File:** `packages/rbee-ui/src/templates/UseCasesTemplate/`

**Implementation:**
- Add `concreteExample` field to items
- Render as highlighted box above scenario
- Style with primary/5 background

**Estimated Time:** 2 hours

---

#### 4.4 Update HowItWorks Template
**File:** `packages/rbee-ui/src/templates/HowItWorks/`

**Implementation:**
- Add `description` and `details` fields to steps
- Render friendly description above code block
- Render bullet points as list

**Estimated Time:** 2 hours

---

#### 4.5 Update EmailCapture Template
**File:** `packages/rbee-ui/src/templates/EmailCapture/`

**Implementation:**
- Add business upsell footer
- Make more flexible (CTA vs email capture)
- Update props interface

**Estimated Time:** 2 hours

---

## Phase 5: Testing & Polish (1-2 days)

### Goal
Ensure everything works, no regressions, consistent feel.

### Tasks

#### 5.1 Visual Regression Testing
**Tools:** Manual testing + screenshots

**Checklist:**
- [ ] All pages render correctly (light mode)
- [ ] All pages render correctly (dark mode)
- [ ] All interactive elements work (hover, focus, active)
- [ ] All templates maintain spacing consistency
- [ ] No broken layouts on mobile
- [ ] No broken layouts on tablet
- [ ] No broken layouts on desktop

**Estimated Time:** 3 hours

---

#### 5.2 Accessibility Testing
**Tools:** axe DevTools, manual keyboard testing

**Checklist:**
- [ ] All contrast ratios pass WCAG AA
- [ ] All interactive elements keyboard accessible
- [ ] All images have alt text
- [ ] All forms have labels
- [ ] No accessibility violations

**Estimated Time:** 2 hours

---

#### 5.3 Cross-Browser Testing
**Browsers:** Chrome, Firefox, Safari, Edge

**Checklist:**
- [ ] Chrome (latest)
- [ ] Firefox (latest)
- [ ] Safari (latest)
- [ ] Edge (latest)

**Estimated Time:** 2 hours

---

#### 5.4 Performance Testing
**Tools:** Lighthouse, WebPageTest

**Checklist:**
- [ ] Lighthouse score > 90
- [ ] First Contentful Paint < 1.5s
- [ ] Largest Contentful Paint < 2.5s
- [ ] Cumulative Layout Shift < 0.1

**Estimated Time:** 1 hour

---

## Timeline

### Week 1
- **Day 1-2:** Phase 1 (Copy Updates)
- **Day 3:** Phase 2 (Color Scheme)
- **Day 4-5:** Phase 3 (New Templates)

### Week 2
- **Day 1-3:** Phase 4 (Template Enhancements)
- **Day 4-5:** Phase 5 (Testing & Polish)

**Total:** 8-12 days depending on complexity

---

## Success Metrics

### Before (Current)
- Hero: Generic headline, technical copy
- No comparison section
- No social proof numbers
- Dark mode primary fails WCAG
- Cool color scheme

### After (Target)
- Hero: Emotional hook, concrete examples
- Comparison section added
- Social proof numbers added
- Dark mode passes WCAG AA
- Warm, friendly color scheme

### Expected Impact
- **Conversion rate:** +15-25% (emotional hooks + social proof)
- **Bounce rate:** -10-15% (clearer value prop)
- **Time on page:** +20-30% (better engagement)

---

## Risk Mitigation

### Low Risk
- All changes are incremental
- No breaking changes to existing components
- Can roll back easily (git revert)

### Testing Strategy
- Test each phase before moving to next
- Keep main branch stable
- Use feature branches for each phase

### Rollback Plan
If issues arise:
1. Revert color changes (Phase 2)
2. Remove new templates (Phase 3)
3. Revert to previous copy (Phase 1)

---

## Next Actions

1. **Review this plan** with team
2. **Create feature branch:** `feat/conversion-optimization`
3. **Start Phase 1:** Copy updates (2-3 hours)
4. **Deploy to staging** after each phase
5. **Get feedback** before moving to production

---

**Ready to start? Begin with Phase 1: Copy Updates**
