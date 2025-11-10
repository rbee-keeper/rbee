# Blog-Specific Components

This document describes the specialized components created for blog posts in the rbee UI library.

## Overview

Four new organism components have been created to support rich, interactive blog content:

1. **BlogCallout** - Enhanced callout boxes
2. **BlogComparisonTable** - Feature/solution comparison tables
3. **BlogCostBreakdown** - ROI and cost analysis
4. **BlogRoadmap** - Product roadmap timelines

## Components

### 1. BlogCallout

Enhanced callout boxes for tips, warnings, examples, and more.

**Location:** `/packages/rbee-ui/src/organisms/BlogCallout/`

**Variants:**
- `info` - Information callouts (💡 blue)
- `success` - Success messages (✅ green)
- `warning` - Warning messages (⚠️ yellow)
- `danger` - Error/danger messages (❌ red)
- `tip` - Pro tips (💡 primary)
- `example` - Example callouts (🚀 default)
- `pricing` - Pricing information (💎 blue)

**Usage:**
```tsx
import { BlogCallout } from '@rbee/ui/organisms'

<BlogCallout variant="tip" title="Pro Tip">
  Use worker metadata to tag workers with custom properties.
</BlogCallout>
```

### 2. BlogComparisonTable

Specialized comparison table for blog posts with column highlighting.

**Location:** `/packages/rbee-ui/src/organisms/BlogComparisonTable/`

**Features:**
- Highlight a specific column (e.g., "rbee" vs competitors)
- Support for boolean values (✓/✗ icons)
- Responsive design with horizontal scrolling
- Hover effects and alternating row colors

**Usage:**
```tsx
import { BlogComparisonTable } from '@rbee/ui/organisms'

<BlogComparisonTable
  title="Deployment Method Comparison"
  columns={['rbee (SSH)', 'Kubernetes', 'Docker Swarm']}
  highlightColumn={0}
  rows={[
    { feature: 'Setup Time', values: ['5 minutes', '2-6 months', '1-2 weeks'] },
    { feature: 'Complexity', values: ['Low', 'Very High', 'Medium'] },
    { feature: 'Multi-GPU Support', values: [true, true, false] },
  ]}
/>
```

### 3. BlogCostBreakdown

ROI and cost analysis component for blog posts.

**Location:** `/packages/rbee-ui/src/organisms/BlogCostBreakdown/`

**Features:**
- Before/After scenarios with trend icons
- Cost items with one-time badges
- Summary section with savings highlighting
- Green-themed success styling

**Usage:**
```tsx
import { BlogCostBreakdown } from '@rbee/ui/organisms'

<BlogCostBreakdown
  title="SaaS Startup Use Case"
  before="Spending $800/month on OpenAI GPT-3.5"
  after="Deployed on 1× RTX 3090 ($800 hardware + €129 license)"
  items={[
    { label: 'Hardware', value: '$800', oneTime: true },
    { label: 'rbee License', value: '€129', oneTime: true },
    { label: 'Power', value: '~$30/month' },
  ]}
  summary={[
    { label: 'Monthly savings', value: '$770', isSavings: true },
    { label: 'Annual savings', value: '$9,240', isSavings: true },
    { label: 'Break-even', value: '1.2 months', isSavings: true },
  ]}
/>
```

### 4. BlogRoadmap

Product roadmap timeline for blog posts.

**Location:** `/packages/rbee-ui/src/organisms/BlogRoadmap/`

**Features:**
- Status indicators (completed, in-progress, planned)
- Color-coded borders and badges
- Emoji support for visual appeal
- Status-based opacity for planned items

**Usage:**
```tsx
import { BlogRoadmap } from '@rbee/ui/organisms'

<BlogRoadmap
  items={[
    {
      milestone: 'M1 (Q1 2026)',
      title: 'Foundation',
      description: 'Core orchestration, chat models, basic GUI',
      status: 'in-progress',
      emoji: '🎯',
    },
    {
      milestone: 'M2 (Q2 2026)',
      title: 'Expansion',
      description: 'Image generation, TTS, premium modules',
      status: 'planned',
      emoji: '🎨',
    },
  ]}
/>
```

## Existing Reusable Components

These existing components are also great for blog posts:

### CodeBlock (Molecule)
```tsx
import { CodeBlock } from '@rbee/ui/molecules'

<CodeBlock
  title="Quick Start Example"
  language="bash"
  code={`# Install rbee
curl -fsSL https://rbee.dev/install.sh | sh`}
  copyable={true}
/>
```

### FeatureInfoCard (Molecule)
```tsx
import { FeatureInfoCard } from '@rbee/ui/molecules'
import { Rocket } from 'lucide-react'

<FeatureInfoCard
  icon={Rocket}
  title="Multi-machine orchestration"
  body="Connect multiple machines via SSH"
  tone="primary"
  variant="compact"
/>
```

### StatsGrid (Molecule)
```tsx
import { StatsGrid } from '@rbee/ui/molecules'

<StatsGrid
  variant="pills"
  columns={3}
  stats={[
    { value: '5 min', label: 'Setup time' },
    { value: '$0', label: 'Monthly cost' },
    { value: '100%', label: 'Data privacy' },
  ]}
/>
```

### TestimonialCard (Molecule)
```tsx
import { TestimonialCard } from '@rbee/ui/molecules'

<TestimonialCard
  name="John Doe"
  role="CTO"
  quote="rbee saved us $10K/month on cloud APIs"
  rating={5}
  verified={true}
/>
```

## Best Practices

1. **Use `not-prose` wrapper** - Wrap components in `<div className="not-prose">` to prevent prose styles from affecting them
2. **Consistent spacing** - Use `my-6` or `my-8` for vertical spacing
3. **Responsive design** - All components are mobile-friendly by default
4. **Dark mode** - All components support dark mode automatically
5. **Accessibility** - Components include proper ARIA labels and semantic HTML

## Example Blog Post Structure

```tsx
<article className="container mx-auto px-4 py-16 max-w-3xl">
  <div className="prose prose-lg dark:prose-invert max-w-none">
    <h2>Problem</h2>
    <p>Description...</p>
    
    <BlogCallout variant="danger" title="Pain Points">
      <ul>...</ul>
    </BlogCallout>
    
    <h2>Solution</h2>
    <div className="grid md:grid-cols-3 gap-4 my-8 not-prose">
      <FeatureInfoCard ... />
      <FeatureInfoCard ... />
      <FeatureInfoCard ... />
    </div>
    
    <h2>Comparison</h2>
    <div className="not-prose my-8">
      <BlogComparisonTable ... />
    </div>
    
    <h2>ROI Analysis</h2>
    <div className="not-prose my-8">
      <BlogCostBreakdown ... />
    </div>
    
    <h2>Roadmap</h2>
    <div className="not-prose">
      <BlogRoadmap ... />
    </div>
  </div>
</article>
```

## Migration Guide

### Old Pattern (Custom HTML/Tailwind)
```tsx
<div className="bg-red-50 dark:bg-red-950 border-l-4 border-red-500 p-6">
  <p className="font-semibold">❌ Pain Points</p>
  <ul>...</ul>
</div>
```

### New Pattern (BlogCallout)
```tsx
<BlogCallout variant="danger" title="Pain Points">
  <ul>...</ul>
</BlogCallout>
```

### Benefits
- ✅ Consistent design system
- ✅ Automatic dark mode support
- ✅ Better accessibility
- ✅ Less code to maintain
- ✅ Reusable across all blog posts
