# Blog Typography Components

Unified, styled typography components for blog posts to ensure consistency and proper styling.

## Components

### 1. BlogHeading

Styled heading component with automatic anchor links and consistent spacing.

**Features:**
- Automatic ID generation from heading text
- Anchor link on hover (#)
- Consistent spacing and sizing
- Border bottom for H2 headings
- Gradient and accent variants

**Usage:**
```tsx
import { BlogHeading } from '@rbee/ui/molecules'

<BlogHeading level="h2">Introduction</BlogHeading>
<BlogHeading level="h3">Getting Started</BlogHeading>
<BlogHeading level="h4" variant="accent">Advanced Topics</BlogHeading>
<BlogHeading level="h2" variant="gradient" id="custom-id">
  Custom Anchor ID
</BlogHeading>
```

**Props:**
- `level`: `'h2' | 'h3' | 'h4' | 'h5' | 'h6'` (default: `'h2'`)
- `variant`: `'default' | 'gradient' | 'accent'` (default: `'default'`)
- `id`: Optional custom ID for anchor links
- `showAnchor`: Show anchor link on hover (default: `true`)

---

### 2. BlogSection

Wrapper component for blog content sections with consistent spacing.

**Features:**
- Consistent vertical spacing
- Optional prose styling
- Semantic section elements
- Anchor link support

**Usage:**
```tsx
import { BlogSection, BlogHeading } from '@rbee/ui/molecules'

<BlogSection>
  <BlogHeading level="h2">Introduction</BlogHeading>
  <p>Some content...</p>
</BlogSection>

{/* For custom components, use noProse */}
<BlogSection noProse>
  <StatsGrid stats={[...]} />
</BlogSection>
```

**Props:**
- `noProse`: Remove prose styling for custom components (default: `false`)
- `id`: Optional section ID for anchor links

---

### 3. BlogList

Styled list component with multiple variants for different use cases.

**Features:**
- Multiple list styles (bullets, numbers, checkmarks, pros/cons, steps)
- Consistent spacing
- Icon support
- Color-coded variants

**Variants:**
- `default`: Standard bullet list
- `ordered`: Numbered list
- `checklist`: Checkmark list (✓)
- `pros`: Green checkmarks for advantages
- `cons`: Red X marks for disadvantages
- `steps`: Chevron arrows for sequential steps

**Usage:**
```tsx
import { BlogList } from '@rbee/ui/molecules'

{/* Checklist */}
<BlogList
  variant="checklist"
  items={[
    'Enable audit logging',
    'Set data retention policies',
    'Implement access controls',
  ]}
/>

{/* Pros */}
<BlogList
  variant="pros"
  items={[
    'Full control over data',
    'No vendor lock-in',
    'Predictable costs',
  ]}
/>

{/* Cons */}
<BlogList
  variant="cons"
  items={[
    'Requires hardware investment',
    'Need technical expertise',
    'Maintenance overhead',
  ]}
/>

{/* Steps */}
<BlogList
  variant="steps"
  spacing="relaxed"
  items={[
    'Install rbee on your machine',
    'Configure your first worker',
    'Download a model',
    'Start making requests',
  ]}
/>
```

**Props:**
- `variant`: List style (see variants above)
- `spacing`: `'compact' | 'default' | 'relaxed'`
- `items`: Array of React nodes to display

---

### 4. BlogDefinitionList

Styled definition list for term/definition pairs, perfect for technical documentation.

**Features:**
- Semantic `<dl>`, `<dt>`, `<dd>` HTML
- Clean term/definition layout
- Multiple styling variants (default, card, highlight)
- Spacing options (default, compact, relaxed)

**Usage:**
```tsx
import { BlogDefinitionList } from '@rbee/ui/molecules'

<BlogDefinitionList
  items={[
    {
      term: <strong>NVIDIA CUDA:</strong>,
      definition: 'Dedicated VRAM (fast, but limited size)',
    },
    {
      term: <strong>Apple Metal:</strong>,
      definition: 'Unified memory (large capacity, shared with CPU)',
    },
  ]}
/>

{/* With card styling */}
<BlogDefinitionList
  itemVariant="card"
  items={[...]}
/>

{/* Compact spacing */}
<BlogDefinitionList
  variant="compact"
  items={[...]}
/>
```

**Props:**
- `items`: Array of `{ term, definition }` objects
- `variant`: `'default' | 'compact' | 'relaxed'` - Spacing between items
- `itemVariant`: `'default' | 'card' | 'highlight'` - Visual style for each item

---

## Complete Blog Post Example

```tsx
import { 
  BlogHeading, 
  BlogSection, 
  BlogList,
  CodeBlock,
  StatsGrid,
  FeatureInfoCard,
  BlogCallout,
} from '@rbee/ui/molecules'

export default function BlogPost() {
  return (
    <article className="container mx-auto px-4 py-16 max-w-3xl">
      <div className="prose prose-lg dark:prose-invert max-w-none">
        
        {/* Main heading */}
        <BlogHeading level="h2">Introduction</BlogHeading>
        <p>Some introductory text...</p>

        {/* Stats section */}
        <BlogSection noProse>
          <StatsGrid
            variant="pills"
            columns={3}
            stats={[
              { value: '5 min', label: 'Setup time' },
              { value: '$0', label: 'Monthly cost' },
              { value: '100%', label: 'Data privacy' },
            ]}
          />
        </BlogSection>

        {/* Subsection with list */}
        <BlogHeading level="h3">Key Benefits</BlogHeading>
        <BlogList
          variant="pros"
          items={[
            'Full control over your data',
            'No vendor lock-in',
            'Predictable costs',
          ]}
        />

        {/* Subsection with checklist */}
        <BlogHeading level="h3">Setup Checklist</BlogHeading>
        <BlogList
          variant="checklist"
          items={[
            'Install rbee CLI',
            'Configure your first worker',
            'Download a model',
            'Test the API',
          ]}
        />

        {/* Code example */}
        <BlogHeading level="h3">Quick Start</BlogHeading>
        <div className="not-prose my-6">
          <CodeBlock
            title="Installation"
            language="bash"
            code={`curl -fsSL https://rbee.dev/install.sh | sh`}
            copyable={true}
          />
        </div>

        {/* Feature cards */}
        <BlogHeading level="h2">Features</BlogHeading>
        <BlogSection noProse>
          <div className="grid md:grid-cols-3 gap-4">
            <FeatureInfoCard
              icon={<Rocket />}
              title="Fast Setup"
              body="Get started in 5 minutes"
              tone="primary"
              variant="compact"
            />
            {/* More cards... */}
          </div>
        </BlogSection>

        {/* Callout */}
        <BlogCallout variant="tip" title="Pro Tip">
          <p className="text-sm">
            Use environment variables to configure your setup.
          </p>
        </BlogCallout>

      </div>
    </article>
  )
}
```

---

## Migration Guide

### Before (Raw HTML)
```tsx
<h2>Introduction</h2>
<p>Some text...</p>

<h3>Key Benefits</h3>
<ul>
  <li>Full control over data</li>
  <li>No vendor lock-in</li>
</ul>
```

### After (Blog Components)
```tsx
<BlogHeading level="h2">Introduction</BlogHeading>
<p>Some text...</p>

<BlogHeading level="h3">Key Benefits</BlogHeading>
<BlogList
  variant="pros"
  items={[
    'Full control over data',
    'No vendor lock-in',
  ]}
/>
```

---

## Benefits

✅ **Consistent styling** - All headings and lists look the same across blog posts  
✅ **Automatic anchor links** - Headings get automatic # links for sharing  
✅ **Better accessibility** - Semantic HTML with proper ARIA labels  
✅ **Dark mode support** - All components work in light and dark themes  
✅ **Less code** - Reusable components instead of repetitive HTML  
✅ **Type safety** - TypeScript props for better DX  

---

## Best Practices

1. **Use BlogHeading for all headings** - Don't use raw `<h2>`, `<h3>`, etc.
2. **Wrap custom components in BlogSection with noProse** - Prevents prose styles from affecting them
3. **Use BlogList for all lists** - Choose the right variant for the content
4. **Keep prose wrapper** - Use `<div className="prose prose-lg dark:prose-invert max-w-none">` for the main content
5. **Use not-prose for components** - Wrap custom components in `<div className="not-prose">`

---

## Component Hierarchy

```
Article
└── prose wrapper
    ├── BlogHeading (h2)
    ├── <p> (regular prose)
    ├── BlogSection (noProse)
    │   └── StatsGrid
    ├── BlogHeading (h3)
    ├── BlogList (pros)
    ├── BlogSection (noProse)
    │   └── CodeBlock
    └── BlogCallout
```
