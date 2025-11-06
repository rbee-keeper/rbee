# TEAM-421: Environment-Aware Presentation Layer

**Date:** 2025-11-06  
**Status:** Analysis & Implementation Plan

---

## The Problem

> "The presentation layer also needs extra SEO on the Next.js site and flexibility in the Tauri GUI. Both for the model and the worker."

**Current State:**
- ❌ Same content rendered in Tauri and Next.js
- ❌ No SEO metadata on Next.js
- ❌ No environment-specific UI elements
- ❌ No flexibility for Tauri-specific features

**What's Needed:**

### Next.js (marketplace.rbee.ai)
- ✅ **SEO metadata** - `<title>`, `<meta>`, Open Graph, Twitter Cards
- ✅ **Structured data** - JSON-LD for search engines
- ✅ **Static content** - Pre-rendered at build time
- ✅ **Marketing content** - "Install rbee to download", feature highlights
- ✅ **Social sharing** - Rich previews on Twitter, Discord, etc.
- ✅ **Search indexing** - Google, Bing, etc.

### Tauri (rbee-keeper desktop app)
- ✅ **Action buttons** - Download, Install, Configure
- ✅ **System integration** - File paths, system info
- ✅ **Real-time data** - Live progress, status updates
- ✅ **Local features** - File browser, settings
- ✅ **No marketing fluff** - Just the functionality

---

## Architecture

### Conditional Rendering Strategy

```tsx
// ArtifactDetailPageTemplate.tsx
import { getEnvironment } from '@rbee/ui/utils';

export function ArtifactDetailPageTemplate({
  // ... existing props
  seoMetadata?: SEOMetadata;  // Next.js only
  extraContent?: ReactNode;   // Environment-specific content
}) {
  const env = getEnvironment();

  return (
    <>
      {/* SEO - Next.js only */}
      {env === 'nextjs-ssg' && seoMetadata && (
        <SEOHead metadata={seoMetadata} />
      )}

      {/* Main content - all environments */}
      <div className="space-y-8">
        {/* Hero header */}
        <header>...</header>

        {/* Environment-specific content */}
        {env === 'nextjs-ssg' && (
          <CallToAction>
            Install rbee to download this model
          </CallToAction>
        )}

        {env === 'tauri' && (
          <SystemInfo>
            Will download to: ~/.cache/rbee/models/
          </SystemInfo>
        )}

        {/* Main content grid */}
        <div className="grid">...</div>

        {/* Extra content slot */}
        {extraContent}
      </div>
    </>
  );
}
```

---

## Implementation Plan

### Phase 1: SEO Components (Next.js)

#### 1.1 SEO Metadata Types
```typescript
// frontend/packages/rbee-ui/src/marketplace/types/seo.ts

export interface SEOMetadata {
  // Basic SEO
  title: string;
  description: string;
  keywords?: string[];
  canonical?: string;

  // Open Graph (Facebook, Discord, etc.)
  ogType?: 'website' | 'article';
  ogImage?: string;
  ogImageAlt?: string;

  // Twitter Card
  twitterCard?: 'summary' | 'summary_large_image';
  twitterImage?: string;

  // Structured Data (JSON-LD)
  structuredData?: Record<string, any>;
}

export interface ModelSEOData extends SEOMetadata {
  modelId: string;
  author: string;
  downloads: number;
  likes: number;
  tags: string[];
  license?: string;
}

export interface WorkerSEOData extends SEOMetadata {
  workerId: string;
  version: string;
  workerType: 'cpu' | 'cuda' | 'metal';
  platforms: string[];
  license: string;
}
```

#### 1.2 SEO Head Component
```tsx
// frontend/packages/rbee-ui/src/marketplace/components/SEOHead.tsx

import { getEnvironment } from '@rbee/ui/utils';

export function SEOHead({ metadata }: { metadata: SEOMetadata }) {
  const env = getEnvironment();

  // Only render in Next.js
  if (env !== 'nextjs-ssg' && env !== 'nextjs-ssr') {
    return null;
  }

  return (
    <>
      {/* Basic SEO */}
      <title>{metadata.title}</title>
      <meta name="description" content={metadata.description} />
      {metadata.keywords && (
        <meta name="keywords" content={metadata.keywords.join(', ')} />
      )}
      {metadata.canonical && (
        <link rel="canonical" href={metadata.canonical} />
      )}

      {/* Open Graph */}
      <meta property="og:title" content={metadata.title} />
      <meta property="og:description" content={metadata.description} />
      <meta property="og:type" content={metadata.ogType || 'website'} />
      {metadata.ogImage && (
        <>
          <meta property="og:image" content={metadata.ogImage} />
          <meta property="og:image:alt" content={metadata.ogImageAlt || metadata.title} />
        </>
      )}

      {/* Twitter Card */}
      <meta name="twitter:card" content={metadata.twitterCard || 'summary_large_image'} />
      <meta name="twitter:title" content={metadata.title} />
      <meta name="twitter:description" content={metadata.description} />
      {metadata.twitterImage && (
        <meta name="twitter:image" content={metadata.twitterImage} />
      )}

      {/* Structured Data */}
      {metadata.structuredData && (
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify(metadata.structuredData),
          }}
        />
      )}
    </>
  );
}
```

#### 1.3 SEO Helper Functions
```typescript
// frontend/packages/rbee-ui/src/marketplace/utils/seo.ts

export function generateModelSEO(model: Model): ModelSEOData {
  return {
    modelId: model.id,
    title: `${model.name} - AI Model | rbee Marketplace`,
    description: `${model.description.substring(0, 160)}... Download ${model.name} by ${model.author}. ${model.downloads.toLocaleString()} downloads, ${model.likes.toLocaleString()} likes.`,
    keywords: [
      model.name,
      model.author,
      'AI model',
      'language model',
      'machine learning',
      ...model.tags,
    ],
    canonical: `https://marketplace.rbee.ai/models/${encodeURIComponent(model.id)}`,
    author: model.author,
    downloads: model.downloads,
    likes: model.likes,
    tags: model.tags,
    license: model.cardData?.license,

    // Open Graph
    ogType: 'article',
    ogImage: `https://marketplace.rbee.ai/api/og/model/${encodeURIComponent(model.id)}`,
    ogImageAlt: `${model.name} - AI Model`,

    // Twitter Card
    twitterCard: 'summary_large_image',
    twitterImage: `https://marketplace.rbee.ai/api/og/model/${encodeURIComponent(model.id)}`,

    // Structured Data (Schema.org)
    structuredData: {
      '@context': 'https://schema.org',
      '@type': 'SoftwareApplication',
      name: model.name,
      description: model.description,
      author: {
        '@type': 'Person',
        name: model.author,
      },
      aggregateRating: {
        '@type': 'AggregateRating',
        ratingValue: '4.5',
        reviewCount: model.likes,
      },
      downloadUrl: `rbee://download-model/${encodeURIComponent(model.id)}`,
      operatingSystem: 'Linux, macOS, Windows',
      applicationCategory: 'DeveloperApplication',
      offers: {
        '@type': 'Offer',
        price: '0',
        priceCurrency: 'USD',
      },
    },
  };
}

export function generateWorkerSEO(worker: WorkerCatalogEntry): WorkerSEOData {
  return {
    workerId: worker.id,
    title: `${worker.name} v${worker.version} - Worker | rbee Marketplace`,
    description: `${worker.description} Supports ${worker.platforms.join(', ')}. Install ${worker.name} for ${worker.workerType.toUpperCase()} acceleration.`,
    keywords: [
      worker.name,
      'rbee worker',
      worker.workerType,
      'AI inference',
      ...worker.platforms,
    ],
    canonical: `https://marketplace.rbee.ai/workers/${worker.id}`,
    version: worker.version,
    workerType: worker.workerType,
    platforms: worker.platforms,
    license: worker.license,

    // Open Graph
    ogType: 'article',
    ogImage: `https://marketplace.rbee.ai/api/og/worker/${worker.id}`,
    ogImageAlt: `${worker.name} - rbee Worker`,

    // Twitter Card
    twitterCard: 'summary',
    twitterImage: `https://marketplace.rbee.ai/api/og/worker/${worker.id}`,

    // Structured Data
    structuredData: {
      '@context': 'https://schema.org',
      '@type': 'SoftwareApplication',
      name: worker.name,
      description: worker.description,
      version: worker.version,
      downloadUrl: `rbee://install-worker/${worker.id}`,
      operatingSystem: worker.platforms.join(', '),
      applicationCategory: 'DeveloperApplication',
      license: worker.license,
    },
  };
}
```

---

### Phase 2: Environment-Specific Content

#### 2.1 Call-to-Action Component (Next.js only)
```tsx
// frontend/packages/rbee-ui/src/marketplace/components/InstallCTA.tsx

import { getEnvironment } from '@rbee/ui/utils';
import { Button, Card } from '@rbee/ui/atoms';
import { Download, ExternalLink } from 'lucide-react';

export function InstallCTA({ artifactType }: { artifactType: 'model' | 'worker' }) {
  const env = getEnvironment();

  // Only show in Next.js (not in Tauri)
  if (env === 'tauri') return null;

  return (
    <Card className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950 dark:to-purple-950">
      <div className="p-6 text-center">
        <Download className="size-12 mx-auto mb-4 text-blue-600" />
        <h3 className="text-xl font-bold mb-2">
          Install rbee to {artifactType === 'model' ? 'download' : 'install'} this {artifactType}
        </h3>
        <p className="text-muted-foreground mb-4">
          rbee is a free, open-source AI orchestration tool that lets you download models
          and install workers directly to your system.
        </p>
        <div className="flex gap-3 justify-center">
          <Button size="lg" asChild>
            <a href="/download">
              <Download className="size-4 mr-2" />
              Download rbee
            </a>
          </Button>
          <Button variant="outline" size="lg" asChild>
            <a href="/docs" target="_blank">
              <ExternalLink className="size-4 mr-2" />
              Learn More
            </a>
          </Button>
        </div>
      </div>
    </Card>
  );
}
```

#### 2.2 System Info Component (Tauri only)
```tsx
// frontend/packages/rbee-ui/src/marketplace/components/SystemInfo.tsx

import { getEnvironment } from '@rbee/ui/utils';
import { Card, CardContent, CardHeader, CardTitle } from '@rbee/ui/atoms';
import { HardDrive, Folder } from 'lucide-react';

export function SystemInfo({ 
  artifactType,
  installPath,
}: { 
  artifactType: 'model' | 'worker';
  installPath: string;
}) {
  const env = getEnvironment();

  // Only show in Tauri (not in Next.js)
  if (env !== 'tauri') return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm flex items-center gap-2">
          <HardDrive className="size-4" />
          System Information
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        <div>
          <span className="text-muted-foreground">Install location:</span>
          <div className="font-mono text-xs mt-1 p-2 bg-muted rounded">
            {installPath}
          </div>
        </div>
        <div className="text-xs text-muted-foreground">
          {artifactType === 'model' 
            ? 'Models are cached locally for faster loading'
            : 'Workers are installed to your system PATH'}
        </div>
      </CardContent>
    </Card>
  );
}
```

---

### Phase 3: Update Templates

#### 3.1 Update ArtifactDetailPageTemplate
```tsx
// frontend/packages/rbee-ui/src/marketplace/templates/ArtifactDetailPageTemplate/ArtifactDetailPageTemplate.tsx

import { SEOHead } from '../../components/SEOHead';
import { InstallCTA } from '../../components/InstallCTA';
import { SystemInfo } from '../../components/SystemInfo';
import { getEnvironment } from '../../../utils/environment';
import type { SEOMetadata } from '../../types/seo';

export interface ArtifactDetailPageTemplateProps {
  // ... existing props

  /** SEO metadata (Next.js only) */
  seoMetadata?: SEOMetadata;

  /** Artifact type for environment-specific content */
  artifactType?: 'model' | 'worker';

  /** Install path (Tauri only) */
  installPath?: string;

  /** Extra environment-specific content */
  extraContent?: ReactNode;
}

export function ArtifactDetailPageTemplate({
  // ... existing props
  seoMetadata,
  artifactType = 'model',
  installPath,
  extraContent,
}: ArtifactDetailPageTemplateProps) {
  const env = getEnvironment();

  return (
    <>
      {/* SEO - Next.js only */}
      {seoMetadata && <SEOHead metadata={seoMetadata} />}

      <div className="space-y-8">
        {/* Back button */}
        {/* ... existing code ... */}

        {/* Hero Header */}
        {/* ... existing code ... */}

        {/* Call-to-Action - Next.js only */}
        {env !== 'tauri' && (
          <InstallCTA artifactType={artifactType} />
        )}

        {/* System Info - Tauri only */}
        {env === 'tauri' && installPath && (
          <SystemInfo 
            artifactType={artifactType}
            installPath={installPath}
          />
        )}

        {/* Main content grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* ... existing code ... */}
        </div>

        {/* Extra environment-specific content */}
        {extraContent}
      </div>
    </>
  );
}
```

#### 3.2 Update ModelDetailPageTemplate
```tsx
// frontend/packages/rbee-ui/src/marketplace/templates/ModelDetailPageTemplate/ModelDetailPageTemplate.tsx

import { generateModelSEO } from '../../utils/seo';
import { getEnvironment } from '../../../utils/environment';

export function ModelDetailPageTemplate({
  model,
  // ... existing props
}: ModelDetailPageTemplateProps) {
  const env = getEnvironment();
  const hfUrl = huggingFaceUrl || `https://huggingface.co/${model.id}`;

  // Generate SEO metadata (Next.js only)
  const seoMetadata = env !== 'tauri' ? generateModelSEO(model) : undefined;

  // Get install path (Tauri only)
  const installPath = env === 'tauri' 
    ? `~/.cache/rbee/models/${model.id.replace('/', '--')}/`
    : undefined;

  // ... rest of component

  return (
    <ArtifactDetailPageTemplate
      name={model.name}
      author={model.author}
      description={model.description}
      // ... existing props
      seoMetadata={seoMetadata}
      artifactType="model"
      installPath={installPath}
      leftSidebar={leftSidebar}
      mainContent={mainContent}
    />
  );
}
```

---

## Benefits

### Next.js (SEO)
✅ **Search Engine Visibility** - Google, Bing index pages  
✅ **Social Sharing** - Rich previews on Twitter, Discord  
✅ **Marketing** - "Install rbee" CTA drives downloads  
✅ **Structured Data** - Better search results  
✅ **Canonical URLs** - Avoid duplicate content  

### Tauri (Flexibility)
✅ **System Integration** - Show local file paths  
✅ **Real-time Info** - Live progress, status  
✅ **No Marketing Fluff** - Just functionality  
✅ **Action Buttons** - Direct download/install  
✅ **Local Features** - File browser, settings  

---

## Example: Model Detail Page

### Next.js (marketplace.rbee.ai)
```html
<!-- SEO metadata -->
<title>Llama-2-7B-Chat - AI Model | rbee Marketplace</title>
<meta name="description" content="Llama 2 7B Chat model by Meta. 7.2M downloads, 760 likes..." />
<meta property="og:image" content="https://marketplace.rbee.ai/api/og/model/..." />
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  "name": "Llama-2-7B-Chat",
  ...
}
</script>

<!-- Page content -->
<div>
  <h1>Llama-2-7B-Chat</h1>
  <p>by Meta</p>

  <!-- CTA Box -->
  <div class="cta-box">
    <h3>Install rbee to download this model</h3>
    <button>Download rbee</button>
  </div>

  <!-- Model details -->
  <div>...</div>
</div>
```

### Tauri (rbee-keeper)
```html
<!-- No SEO metadata (not needed) -->

<!-- Page content -->
<div>
  <h1>Llama-2-7B-Chat</h1>
  <p>by Meta</p>

  <!-- System Info Box -->
  <div class="system-info">
    <h4>System Information</h4>
    <p>Install location:</p>
    <code>~/.cache/rbee/models/meta-llama--Llama-2-7B-Chat/</code>
  </div>

  <!-- Model details -->
  <div>...</div>

  <!-- Action buttons -->
  <button>Download Model</button>
</div>
```

---

## Implementation Checklist

### Phase 1: SEO (Next.js)
- [ ] Create `types/seo.ts` with SEO types
- [ ] Create `components/SEOHead.tsx` component
- [ ] Create `utils/seo.ts` with helper functions
- [ ] Update `ModelDetailPageTemplate` to use SEO
- [ ] Update `WorkerDetailPageTemplate` to use SEO (when created)
- [ ] Test SEO metadata in Next.js build
- [ ] Verify Open Graph previews

### Phase 2: Environment-Specific Content
- [ ] Create `components/InstallCTA.tsx` (Next.js only)
- [ ] Create `components/SystemInfo.tsx` (Tauri only)
- [ ] Update `ArtifactDetailPageTemplate` with conditional rendering
- [ ] Test CTA shows in Next.js, not in Tauri
- [ ] Test SystemInfo shows in Tauri, not in Next.js

### Phase 3: Integration
- [ ] Update `ModelDetailPageTemplate` with all features
- [ ] Update `WorkerDetailsPage` with all features
- [ ] Test in Tauri environment
- [ ] Test in Next.js environment
- [ ] Verify SEO metadata
- [ ] Verify conditional content

---

## Success Metrics

✅ **SEO:** Google indexes pages, rich previews work  
✅ **Tauri:** System info shows, no marketing fluff  
✅ **Next.js:** CTA shows, drives rbee downloads  
✅ **Consistency:** Same data, different presentation  
✅ **Flexibility:** Easy to add environment-specific features  

---

## Next Steps

1. **HIGH:** Implement SEO components and helpers
2. **HIGH:** Implement environment-specific content components
3. **MEDIUM:** Update templates to use new components
4. **MEDIUM:** Test in both environments
5. **LOW:** Add Open Graph image generation API

**Start with Phase 1 - SEO components!**
