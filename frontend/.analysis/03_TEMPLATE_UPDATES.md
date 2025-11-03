# Template & Component Recommendations

**Date:** November 3, 2025  
**Goal:** Identify which templates need updates and which new templates to create

---

## New Templates Needed

### 1. ComparisonCardTemplate (HIGH PRIORITY)

**IMPORTANT:** We already have a `ComparisonTemplate` for feature matrices (table-based). This is a DIFFERENT component for visual "before/after" comparison.

**Purpose:** "Without rbee vs With rbee" visual comparison section

**Reference Design:**
```
Two side-by-side cards:
- Left: Red-tinted card with X marks (problems)
- Right: Green-tinted card with checkmarks (solutions)

Each card has:
- Icon badge (X or checkmark)
- Title ("Without rbee" / "With rbee")
- 5 bullet points with icon + title + description
```

**Naming:**
- Keep existing: `ComparisonTemplate` (feature matrix table)
- New component: `BeforeAfterTemplate` or `VisualComparisonTemplate`
- Avoids naming confusion

**Props Interface:**
```typescript
interface ComparisonCardTemplateProps {
  title: string;
  description?: string;
  leftCard: {
    title: string;
    icon: ReactNode;
    items: Array<{
      title: string;
      description: string;
    }>;
    tone: 'destructive';
  };
  rightCard: {
    title: string;
    icon: ReactNode;
    items: Array<{
      title: string;
      description: string;
    }>;
    tone: 'success' | 'primary';
  };
  background?: TemplateBackgroundProps;
}
```

**Location:** `packages/rbee-ui/src/templates/BeforeAfterTemplate/`

**Why it's needed:**
- Reference site has this visual comparison, we only have matrix table
- Powerful visual contrast (red X vs green checkmark)
- Clearly shows problem → solution transformation
- High conversion impact
- Complements existing ComparisonTemplate (which shows feature matrices)

---

### 2. StatsBarTemplate (MEDIUM PRIORITY)

**Purpose:** Social proof numbers section

**Reference Design:**
```
Horizontal bar with 3-4 stat cards:
- Large number (e.g., "5,000+")
- Label (e.g., "Active Users")
- Centered layout
- Subtle background
```

**Props Interface:**
```typescript
interface StatsBarTemplateProps {
  stats: Array<{
    value: string;
    label: string;
    icon?: ReactNode;
  }>;
  background?: TemplateBackgroundProps;
  layout?: 'centered' | 'spread';
}
```

**Location:** `packages/rbee-ui/src/templates/StatsBarTemplate/`

**Why it's needed:**
- Social proof is critical for conversion
- Reference uses this effectively
- We have StatsGrid molecule but no template wrapper
- Easy to implement

---

## Existing Templates to Update

### 1. HeroTemplate (CRITICAL)

**Current Issues:**
- Badge text is too technical ("100% Open Source • GPL-3.0-or-later")
- No social proof in hero
- Subcopy is too technical

**Recommended Updates:**

#### Badge
```typescript
// BEFORE
badge: {
  variant: 'simple',
  text: '100% Open Source • GPL-3.0-or-later',
}

// AFTER
badge: {
  variant: 'simple',
  text: 'Free Forever • No Credit Card • No Limits',
}
```

#### Headline
```typescript
// BEFORE
headline: {
  variant: 'two-line-highlight',
  prefix: 'AI Infrastructure.',
  highlight: 'On Your Terms.',
}

// AFTER (Option 1 - Direct)
headline: {
  variant: 'two-line-highlight',
  prefix: 'Stop Paying for AI APIs.',
  highlight: 'Run Everything Free.',
}

// AFTER (Option 2 - Benefit-focused)
headline: {
  variant: 'two-line-highlight',
  prefix: 'Your GPUs. Zero API Fees.',
  highlight: 'One Simple API.',
}
```

#### Subcopy
```typescript
// BEFORE
subcopy: 'rbee (pronounced "are-bee") is your OpenAI-compatible AI stack. Run LLMs on **your** hardware across every GPU and machine. Build with AI, keep control, and escape provider lock-in.'

// AFTER
subcopy: 'Run ChatGPT and Stable Diffusion simultaneously on your hardware. Fine-tune models while generating images. Zero conflicts. Zero cost. Your Mac, PC, and Linux GPUs working together like a coordinated swarm.'
```

#### Add Social Proof (SKIP FOR NOW)
**Note:** rbee is currently a solo project (~7 stars). Do not add fake social proof numbers.

```typescript
// SKIP THIS - No social proof yet
// When you have real community, add:
// socialProof: {
//   text: 'Early stage - actively building',
//   items: [
//     { icon: 'github', label: 'Open source (GPL-3.0)' },
//     { icon: 'check', label: '68% complete (v0.1.0)' },
//   ],
// }
```

#### Add Quantified Benefits (Optional)
```typescript
// NEW ADDITION - Based on reference business page
quantifiedBenefits: {
  items: [
    { metric: '40-60%', label: 'Higher GPU utilization' },
    { metric: '$240-1,200', label: 'Saved per year' },
    { metric: '5 min', label: 'Setup time' },
  ],
}
```

**Template Changes Needed:**
- Add optional `socialProof` prop to HeroTemplateProps
- Add optional `quantifiedBenefits` prop for specific ROI metrics
- Render social proof below CTAs
- Render quantified benefits as small stat pills
- Style as small badges or inline text

---

### 2. WhatIsRbee Template (HIGH)

**Current Issues:**
- Title is a question ("What is rbee?")
- Description is too technical
- Doesn't emphasize transformation

**Recommended Updates:**

#### Title
```typescript
// BEFORE
title: 'What is rbee?'

// AFTER (Option 1 - Transformation)
title: 'Turn Scattered GPUs Into One Intelligent Swarm'

// AFTER (Option 2 - Problem-solution)
title: 'Stop Juggling AI Tools. One API for Everything.'
```

#### Description
```typescript
// BEFORE
description: 'is an open-source AI orchestrator that unifies every computer in your home or office into one OpenAI-compatible cluster - private, controllable, and yours.'

// AFTER
description: 'Your GPUs are powerful but scattered. rbee transforms them into an intelligent swarm. One queen orchestrates. Multiple workers execute. You get one simple API that just works—OpenAI-compatible, private, and yours.'
```

#### Add Bee Metaphor Emphasis
```typescript
// NEW ADDITION
metaphor: {
  icon: <BeeIcon />,
  text: 'Like a bee colony: one queen coordinates, multiple workers execute',
}
```

**Template Changes Needed:**
- Update WhatIsRbeeProps to support transformation-focused copy
- Add optional metaphor visualization
- Consider renaming template to `TransformationTemplate`

---

### 3. UseCasesTemplate (HIGH)

**Current Issues:**
- Title is abstract ("Built for those who value independence")
- Persona-based instead of action-based
- Doesn't emphasize "impossible → possible"

**Recommended Updates:**

#### Title & Description
```typescript
// BEFORE
title: 'Built for those who value independence'
description: 'Run serious AI on your own hardware. Keep costs at zero, keep control at 100%.'

// AFTER
title: 'Do What Was Impossible Before'
description: 'Remember when running two AI models crashed your system? Those days are over. Multitask with your GPUs like never before.'
```

#### Use Case Items - Add Concrete Examples
```typescript
// BEFORE (Abstract)
{
  icon: <Laptop />,
  title: 'The solo developer',
  scenario: 'Shipping a SaaS with AI features; wants control without vendor lock-in.',
  solution: 'Run rbee on your gaming PC + spare workstation...',
  outcome: '$0/month AI costs. Full control. No rate limits.',
}

// AFTER (Concrete + Abstract)
{
  icon: <Laptop />,
  title: 'The solo developer',
  concreteExample: 'Run Cursor AI + Stable Diffusion simultaneously—code and design without switching tools',
  scenario: 'Shipping a SaaS with AI features; wants control without vendor lock-in.',
  solution: 'Run rbee on your gaming PC + spare workstation. Llama 70B for coding, SD for assets—local & fast.',
  outcome: '$0/month AI costs. Full control. No rate limits.',
}
```

**Template Changes Needed:**
- Add `concreteExample` field to use case items
- Render concrete example prominently (bold or highlighted)
- Keep existing persona-based content below

---

### 4. HowItWorks Template (MEDIUM)

**Current Issues:**
- Says "15 minutes" instead of "5 minutes"
- Too technical (shows raw terminal commands)
- Doesn't emphasize "no friction"

**Recommended Updates:**

#### Title & Description
```typescript
// BEFORE
title: 'From zero to AI infrastructure in 15 minutes'

// AFTER
title: 'Up and Running in 5 Minutes'
description: 'No complex configuration. No PhD required. No payment information. Just download, install, and start orchestrating.'
```

#### Step Descriptions - Friendlier Language
```typescript
// BEFORE
{
  label: 'Install rbee',
  block: { kind: 'terminal', ... }
}

// AFTER
{
  label: 'Install rbee on Your Machines',
  description: 'Simple installation on Mac, PC, or Linux. One command gets you started. Works with your existing setup.',
  details: ['brew install rbee', 'apt-get install rbee', 'Or download from GitHub'],
  block: { kind: 'terminal', ... }
}
```

**Template Changes Needed:**
- Add `description` field to steps
- Add `details` array for bullet points
- Render details as friendly list before code block

---

### 5. EmailCapture Template (CRITICAL)

**Current Issues:**
- Shows development status ("In Development · M0 · 68%")
- Uses "Join Waitlist" instead of "Download"
- Creates doubt about readiness

**Recommended Updates:**

#### Badge
```typescript
// BEFORE
badge: {
  text: 'In Development · M0 · 68%',
  showPulse: true,
}

// AFTER
badge: {
  text: 'Free Forever • Others Charge $99/month • You Pay $0',
  showPulse: true,
}
```

#### Headline & Subheadline
```typescript
// BEFORE
headline: 'Get Updates. Own Your AI.'
subheadline: 'Join the rbee waitlist for early access, build notes, and launch perks for running AI on your hardware.'

// AFTER
headline: 'Start Orchestrating Your GPUs in 5 Minutes'
subheadline: 'While others charge hundreds per month for GPU orchestration, rbee is 100% free. No trials. No credit cards. No surprises. Download now and transform your GPU workflow forever.'
```

#### CTA Button
```typescript
// BEFORE
submitButton: {
  label: 'Join Waitlist',
}

// AFTER (Option 1 - Direct download)
submitButton: {
  label: 'Download Free Now',
  href: '/download',
}

// AFTER (Option 2 - Keep email capture)
submitButton: {
  label: 'Get Started Free',
}
```

#### Add Business Upsell
```typescript
// NEW ADDITION
footer: {
  text: 'Need enterprise features?',
  linkText: 'Get Rhai scheduler, telemetry, and GDPR compliance',
  linkHref: '/enterprise',
}
```

**Template Changes Needed:**
- Make EmailCapture more flexible (can be CTA or email capture)
- Add optional business upsell footer
- Support direct download CTA

---

## Component Updates Needed

### 1. Badge Component (MINOR)

**Add Pulse Animation Variant:**
```typescript
// Current
<Badge variant="simple" text="..." />

// Needed
<Badge variant="simple" text="..." showPulse={true} />
```

**Implementation:**
- Add `showPulse` prop
- Add pulsing dot animation (like reference)
- Use for "Free Forever" badge

---

### 2. StatsGrid Molecule (MINOR)

**Current:** Works well, but needs template wrapper

**Recommendation:**
- Keep molecule as-is
- Create StatsBarTemplate wrapper (see above)

---

### 3. HexagonCard (REFERENCE ONLY)

**Reference has HexagonCard component we don't have:**
```typescript
<HexagonCard icon={<Icon />}>
  <h3>Title</h3>
  <p>Description</p>
  <ul>Bullet points</ul>
</HexagonCard>
```

**Do we need it?**
- ❌ No - our IconCard + Card components cover this
- ✅ Reference design is nice, but not critical
- ⏭️ Consider for future visual refresh

---

## Template Consolidation Opportunities

### Merge Similar Templates

#### Hero Templates (6 → 1)
- ✅ Already done - all use HeroTemplate
- Just need to update props

#### CTA Templates (3 → 1)
- EmailCapture
- CTATemplate
- EnterpriseCTA

**Recommendation:**
- Keep EmailCapture for email-specific features
- Merge CTATemplate + EnterpriseCTA into single flexible CTATemplate
- Add variants: 'email-capture' | 'direct-cta' | 'business-upsell'

---

## Implementation Priority

### Phase 1: Critical Copy Updates (1-2 days)
1. ✅ Update HeroTemplate props (headline, badge, subcopy)
2. ✅ Update EmailCapture props (remove "In Development")
3. ✅ Update WhatIsRbee props (transformation title)
4. ✅ Update UseCasesTemplate props (concrete examples)

**Impact:** Immediate conversion improvement, no template changes needed

### Phase 2: New Templates (2-3 days)
1. ✅ Create BeforeAfterTemplate (visual without vs with) - RENAMED to avoid confusion with existing ComparisonTemplate
2. ✅ Create StatsBarTemplate (social proof)
3. ✅ Wire into HomePage

**Note:** Keep existing ComparisonTemplate (feature matrix table) - it's different from the visual before/after comparison

**Impact:** Add missing high-conversion sections

### Phase 3: Template Enhancements (2-3 days)
1. ✅ Add socialProof prop to HeroTemplate
2. ✅ Add concreteExample field to UseCasesTemplate
3. ✅ Add description/details to HowItWorks steps
4. ✅ Add business upsell to EmailCapture

**Impact:** Polish existing templates with reference features

### Phase 4: Component Polish (1-2 days)
1. ✅ Add pulse animation to Badge
2. ✅ Test all color updates
3. ✅ Ensure consistency across all templates

**Impact:** Final polish, cohesive feel

---

## Template Architecture Principles

### Keep Current Strengths
1. ✅ **Reusable templates** - Don't break existing pattern
2. ✅ **Props-based configuration** - Keep declarative approach
3. ✅ **Consistent spacing** - Maintain TemplateContainer pattern
4. ✅ **Atomic design** - Keep atoms → molecules → templates hierarchy

### Add Reference Strengths
1. ✅ **Emotional hooks** - Update copy to create urgency
2. ✅ **Concrete examples** - Add specific use cases
3. ✅ **Visual contrast** - Add comparison template
4. ✅ **Social proof** - Add stats bar template

### Don't Copy Blindly
1. ❌ **HexagonCard** - Our IconCard is fine
2. ❌ **Inline SVG icons** - We use lucide-react (better)
3. ❌ **Custom hexagon patterns** - Our decorations work

---

## Next Steps

1. ✅ Copy analysis complete (01_COPY_ANALYSIS.md)
2. ✅ Color analysis complete (02_COLOR_ANALYSIS.md)
3. ✅ Template recommendations complete (03_TEMPLATE_UPDATES.md)
4. ⏭️ Create component updates (04_COMPONENT_CHANGES.md)
5. ⏭️ Create implementation plan (05_IMPLEMENTATION_PLAN.md)
