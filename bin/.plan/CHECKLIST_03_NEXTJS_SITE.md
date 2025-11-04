# Checklist 03: Next.js Marketplace Site

**Timeline:** 1 week  
**Status:** 📋 NOT STARTED  
**Dependencies:** Checklist 01 (Shared Components), Checklist 02 (SDK)

---

## 🎯 Goal

Build `marketplace.rbee.dev` with Next.js SSG, pre-render top 1000 models, implement installation-aware buttons.

---

## 📦 Phase 1: Project Setup (Day 1)

### 1.1 Hook Up Workspace Packages

- [ ] Navigate to marketplace: `cd frontend/apps/marketplace`
- [ ] Update `package.json` to use workspace packages:
  ```json
  {
    "name": "marketplace",
    "version": "0.1.0",
    "private": true,
    "scripts": {
      "dev": "next dev",
      "build": "next build",
      "start": "next start",
      "lint": "next lint",
      "deploy": "opennextjs-cloudflare build && opennextjs-cloudflare deploy",
      "preview": "opennextjs-cloudflare build && opennextjs-cloudflare preview",
      "cf-typegen": "wrangler types --env-interface CloudflareEnv ./cloudflare-env.d.ts"
    },
    "dependencies": {
      "@opennextjs/cloudflare": "^1.3.0",
      "@rbee/ui": "workspace:*",
      "next": "15.4.6",
      "react": "19.1.0",
      "react-dom": "19.1.0"
    },
    "devDependencies": {
      "@repo/eslint-config": "workspace:*",
      "@repo/tailwind-config": "workspace:*",
      "@repo/typescript-config": "workspace:*",
      "@eslint/eslintrc": "^3",
      "@tailwindcss/postcss": "^4",
      "@types/node": "^20.19.24",
      "@types/react": "^19",
      "@types/react-dom": "^19",
      "eslint": "^9",
      "eslint-config-next": "15.4.6",
      "tailwindcss": "^4",
      "typescript": "^5",
      "wrangler": "^4.45.3"
    }
  }
  ```
- [ ] Install dependencies: `pnpm install`
- [ ] Verify dev server works: `pnpm dev`

### 1.2 Configure TypeScript

- [ ] Update `tsconfig.json` to extend workspace config:
  ```json
  {
    "extends": "@repo/typescript-config/react-app.json",
    "compilerOptions": {
      "target": "ES2017",
      "lib": ["dom", "dom.iterable", "esnext"],
      "allowJs": true,
      "skipLibCheck": true,
      "strict": true,
      "noEmit": true,
      "esModuleInterop": true,
      "module": "esnext",
      "moduleResolution": "bundler",
      "resolveJsonModule": true,
      "isolatedModules": true,
      "jsx": "preserve",
      "incremental": true,
      "plugins": [
        {
          "name": "next"
        }
      ],
      "paths": {
        "@/*": ["./*"]
      }
    },
    "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
    "exclude": ["node_modules"]
  }
  ```
- [ ] Verify TypeScript works: `pnpm run typecheck` (add script if needed)

### 1.3 Configure Tailwind

- [ ] Create `tailwind.config.ts`:
  ```typescript
  import type { Config } from 'tailwindcss'
  
  const config: Config = {
    content: [
      './app/**/*.{js,ts,jsx,tsx,mdx}',
      './components/**/*.{js,ts,jsx,tsx,mdx}',
      '../../packages/rbee-ui/src/**/*.{js,ts,jsx,tsx}'
    ],
    theme: {
      extend: {}
    },
    plugins: []
  }
  
  export default config
  ```
- [ ] Create `postcss.config.mjs`:
  ```javascript
  export default {
    plugins: {
      '@tailwindcss/postcss': {},
    },
  }
  ```
- [ ] Update `app/globals.css`:
  ```css
  @import "@rbee/ui/globals";
  @import "tailwindcss";
  ```
- [ ] Test styles work: `pnpm dev`

### 1.4 Configure ESLint

- [ ] Create `eslint.config.mjs`:
  ```javascript
  import { dirname } from "path";
  import { fileURLToPath } from "url";
  import { FlatCompat } from "@eslint/eslintrc";

  const __filename = fileURLToPath(import.meta.url);
  const __dirname = dirname(__filename);

  const compat = new FlatCompat({
    baseDirectory: __dirname,
  });

  const eslintConfig = [
    ...compat.extends(
      "next/core-web-vitals",
      "next/typescript",
      "@repo/eslint-config/react"
    ),
  ];

  export default eslintConfig;
  ```
- [ ] Test linting: `pnpm lint`

### 1.5 Configure Next.js

- [ ] Update `next.config.ts`:
  ```typescript
  import type { NextConfig } from "next";
  import { setupDevPlatform } from '@cloudflare/next-on-pages/next-dev';

  if (process.env.NODE_ENV === 'development') {
    await setupDevPlatform();
  }

  const nextConfig: NextConfig = {
    images: {
      unoptimized: true,
      remotePatterns: [
        {
          protocol: 'https',
          hostname: 'huggingface.co',
        },
        {
          protocol: 'https',
          hostname: 'cdn-lfs.huggingface.co',
        },
        {
          protocol: 'https',
          hostname: 'image.civitai.com',
        },
      ],
    },
    transpilePackages: ['@rbee/ui'],
  };

  export default nextConfig;
  ```
- [ ] Test build: `pnpm build`
- [ ] Verify Cloudflare build works

### 1.6 Create Layout with @rbee/ui Components

- [ ] Update `app/layout.tsx`:
  ```tsx
  import type { Metadata } from 'next'
  import { GeistSans } from 'geist/font/sans'
  import { GeistMono } from 'geist/font/mono'
  import './globals.css'
  
  export const metadata: Metadata = {
    title: 'rbee Marketplace - Run AI Models Locally',
    description: 'Browse and download AI models. Run them locally with rbee. Free, private, unlimited.',
    keywords: ['AI', 'LLM', 'Stable Diffusion', 'local AI', 'rbee'],
    openGraph: {
      title: 'rbee Marketplace',
      description: 'Run AI models locally. Free, private, unlimited.',
      url: 'https://marketplace.rbee.dev',
      siteName: 'rbee Marketplace',
      type: 'website'
    }
  }
  
  export default function RootLayout({
    children
  }: {
    children: React.ReactNode
  }) {
    return (
      <html lang="en" className={`${GeistSans.variable} ${GeistMono.variable}`}>
        <body className="font-sans antialiased">
          <nav className="border-b">
            <div className="container mx-auto px-4 py-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-2xl">🐝</span>
                  <span className="font-bold text-xl">rbee Marketplace</span>
                </div>
                <div className="flex items-center gap-4">
                  <a href="/models" className="hover:underline">Models</a>
                  <a href="/workers" className="hover:underline">Workers</a>
                  <a href="https://rbee.dev" className="hover:underline">About</a>
                </div>
              </div>
            </div>
          </nav>
          <main>{children}</main>
          <footer className="border-t mt-16">
            <div className="container mx-auto px-4 py-8 text-center text-sm text-muted-foreground">
              <p>rbee Marketplace - Run AI models locally. Free, private, unlimited.</p>
            </div>
          </footer>
        </body>
      </html>
    )
  }
  ```
- [ ] Test layout renders correctly

---

## 🏠 Phase 2: Home Page (Day 2)

### 2.1 Create Home Page

- [ ] Update `app/page.tsx`:
  ```tsx
  export default function HomePage() {
    return (
      <div className="container mx-auto px-4 py-16">
        <div className="text-center max-w-3xl mx-auto">
          <h1 className="text-5xl font-bold mb-6">
            Run AI Models Locally
          </h1>
          <p className="text-xl text-gray-600 mb-8">
            Browse thousands of AI models. Download and run them on your own hardware.
            Free, private, unlimited.
          </p>
          
          <div className="flex gap-4 justify-center">
            <a 
              href="/models"
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              Browse Models
            </a>
            <a 
              href="/workers"
              className="px-6 py-3 border border-gray-300 rounded-lg hover:bg-gray-50"
            >
              Browse Workers
            </a>
          </div>
        </div>
        
        <div className="grid md:grid-cols-3 gap-8 mt-16">
          <div className="text-center">
            <div className="text-4xl mb-4">🤗</div>
            <h3 className="font-bold mb-2">HuggingFace Models</h3>
            <p className="text-gray-600">
              Browse thousands of LLM models from HuggingFace
            </p>
          </div>
          
          <div className="text-center">
            <div className="text-4xl mb-4">🎨</div>
            <h3 className="font-bold mb-2">CivitAI Models</h3>
            <p className="text-gray-600">
              Discover image generation models from CivitAI
            </p>
          </div>
          
          <div className="text-center">
            <div className="text-4xl mb-4">👷</div>
            <h3 className="font-bold mb-2">Worker Binaries</h3>
            <p className="text-gray-600">
              Install optimized workers for your hardware
            </p>
          </div>
        </div>
      </div>
    )
  }
  ```
- [ ] Test home page renders
- [ ] Test links work

---

## 📦 Phase 3: Models List Page (Day 3)

### 3.1 Create Models Page

- [ ] Create `app/models/page.tsx`:
  ```tsx
  import { HuggingFaceClient } from '@rbee/marketplace-sdk'
  import { ModelCard, MarketplaceGrid } from '@rbee/marketplace-components'
  
  export const metadata = {
    title: 'AI Models - rbee Marketplace',
    description: 'Browse and download AI models from HuggingFace and CivitAI'
  }
  
  export default async function ModelsPage() {
    // Fetch models at build time (SSG)
    const hfClient = new HuggingFaceClient()
    const models = await hfClient.listModels({ limit: 100, sort: 'popular' })
    
    return (
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold mb-8">AI Models</h1>
        
        <MarketplaceGrid
          items={models}
          renderItem={(model) => (
            <ModelCard
              key={model.id}
              model={model}
              downloadButton={
                <a 
                  href={`/models/${encodeURIComponent(model.id)}`}
                  className="btn-primary"
                >
                  View Details
                </a>
              }
              mode="nextjs"
            />
          )}
        />
      </div>
    )
  }
  
  // Revalidate every hour
  export const revalidate = 3600
  ```
- [ ] Test page renders
- [ ] Test SSG works: `pnpm build`
- [ ] Verify models appear

### 3.2 Add Search & Filter

- [ ] Create `app/models/page.tsx` with client component:
  ```tsx
  'use client'
  
  import { useState, useEffect } from 'react'
  import { HuggingFaceClient } from '@rbee/marketplace-sdk'
  import { ModelCard, MarketplaceGrid, SearchBar } from '@rbee/marketplace-components'
  
  export default function ModelsPage() {
    const [models, setModels] = useState([])
    const [isLoading, setIsLoading] = useState(true)
    const [search, setSearch] = useState('')
    
    useEffect(() => {
      const client = new HuggingFaceClient()
      client.listModels({ search, limit: 100 })
        .then(setModels)
        .finally(() => setIsLoading(false))
    }, [search])
    
    return (
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold mb-8">AI Models</h1>
        
        <SearchBar
          value={search}
          onChange={setSearch}
          placeholder="Search models..."
        />
        
        <MarketplaceGrid
          items={models}
          isLoading={isLoading}
          renderItem={(model) => (
            <ModelCard
              key={model.id}
              model={model}
              downloadButton={
                <a 
                  href={`/models/${encodeURIComponent(model.id)}`}
                  className="btn-primary"
                >
                  View Details
                </a>
              }
              mode="nextjs"
            />
          )}
        />
      </div>
    )
  }
  ```
- [ ] Test search works
- [ ] Test loading state

---

## 📄 Phase 4: Model Detail Page (Day 4)

### 4.1 Create Dynamic Route

- [ ] Create `app/models/[id]/page.tsx`:
  ```tsx
  import { HuggingFaceClient } from '@rbee/marketplace-sdk'
  import { ModelCard } from '@rbee/marketplace-components'
  import { InstallationAwareButton } from '@/components/InstallationAwareButton'
  import type { Metadata } from 'next'
  
  interface Props {
    params: { id: string }
  }
  
  // Generate static paths for top 1000 models
  export async function generateStaticParams() {
    const client = new HuggingFaceClient()
    const models = await client.listModels({ limit: 1000, sort: 'popular' })
    
    return models.map(model => ({
      id: encodeURIComponent(model.id)
    }))
  }
  
  // Generate metadata for SEO
  export async function generateMetadata({ params }: Props): Promise<Metadata> {
    const client = new HuggingFaceClient()
    const model = await client.getModel(decodeURIComponent(params.id))
    
    return {
      title: `${model.name} - rbee Marketplace`,
      description: `${model.description} | Run ${model.name} locally with rbee. Free, private, unlimited.`,
      keywords: [model.name, 'rbee', 'AI model', 'local AI', ...model.tags],
      openGraph: {
        title: model.name,
        description: model.description,
        images: model.imageUrl ? [model.imageUrl] : []
      }
    }
  }
  
  export default async function ModelDetailPage({ params }: Props) {
    const client = new HuggingFaceClient()
    const model = await client.getModel(decodeURIComponent(params.id))
    
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          <ModelCard
            model={model}
            downloadButton={
              <InstallationAwareButton
                modelId={model.id}
                modelName={model.name}
              />
            }
            mode="nextjs"
          />
          
          <div className="mt-8 prose max-w-none">
            <h2>About {model.name}</h2>
            <p>{model.description}</p>
            
            <h3>How to use with rbee</h3>
            <ol>
              <li>Install rbee Keeper</li>
              <li>Click "Run with rbee"</li>
              <li>Model downloads and runs automatically</li>
            </ol>
            
            <h3>Why use rbee?</h3>
            <ul>
              <li>✅ Free forever - no API costs</li>
              <li>✅ 100% private - your data never leaves your machine</li>
              <li>✅ No limits - run as much as you want</li>
              <li>✅ Use your own GPU - maximize performance</li>
            </ul>
          </div>
        </div>
      </div>
    )
  }
  
  export const revalidate = 3600
  ```
- [ ] Test page renders for specific model
- [ ] Test SSG generates 1000 pages
- [ ] Verify metadata is correct

### 4.2 Create Installation-Aware Button

- [ ] Create `components/InstallationAwareButton.tsx`:
  ```tsx
  'use client'
  
  import { useState } from 'react'
  import { openInKeeperWithIframe } from '@/lib/protocolDetection'
  import { InstallModal } from './InstallModal'
  
  interface Props {
    modelId: string
    modelName: string
  }
  
  export function InstallationAwareButton({ modelId, modelName }: Props) {
    const [showInstallModal, setShowInstallModal] = useState(false)
    const [isChecking, setIsChecking] = useState(false)
    
    const handleClick = async () => {
      setIsChecking(true)
      
      const rbeeUrl = `rbee://download/model/huggingface/${modelId}`
      const opened = await openInKeeperWithIframe(rbeeUrl)
      
      setIsChecking(false)
      
      if (!opened) {
        setShowInstallModal(true)
      }
    }
    
    return (
      <>
        <button 
          onClick={handleClick}
          disabled={isChecking}
          className="w-full px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
        >
          {isChecking ? '⏳ Opening rbee...' : `🚀 Run ${modelName} with rbee`}
        </button>
        
        {showInstallModal && (
          <InstallModal
            onClose={() => setShowInstallModal(false)}
            modelId={modelId}
            modelName={modelName}
          />
        )}
      </>
    )
  }
  ```
- [ ] Test button click
- [ ] Test protocol detection

### 4.3 Create Protocol Detection

- [ ] Create `lib/protocolDetection.ts`:
  ```typescript
  export async function openInKeeperWithIframe(url: string): Promise<boolean> {
    return new Promise((resolve) => {
      const iframe = document.createElement('iframe')
      iframe.style.display = 'none'
      document.body.appendChild(iframe)
      
      let timeout: NodeJS.Timeout
      let hasBlurred = false
      
      const handleBlur = () => {
        hasBlurred = true
        clearTimeout(timeout)
        cleanup()
        resolve(true)
      }
      
      const cleanup = () => {
        window.removeEventListener('blur', handleBlur)
        document.body.removeChild(iframe)
      }
      
      timeout = setTimeout(() => {
        if (!hasBlurred) {
          cleanup()
          resolve(false)
        }
      }, 2000)
      
      window.addEventListener('blur', handleBlur)
      iframe.src = url
    })
  }
  ```
- [ ] Test detection works

### 4.4 Create Install Modal

- [ ] Create `components/InstallModal.tsx`:
  ```tsx
  'use client'
  
  interface Props {
    onClose: () => void
    modelId: string
    modelName: string
  }
  
  export function InstallModal({ onClose, modelId, modelName }: Props) {
    const rbeeUrl = `rbee://download/model/huggingface/${modelId}`
    
    return (
      <div 
        className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
        onClick={onClose}
      >
        <div 
          className="bg-white rounded-lg p-8 max-w-2xl w-full mx-4"
          onClick={(e) => e.stopPropagation()}
        >
          <h2 className="text-2xl font-bold mb-4">Install rbee Keeper</h2>
          <p className="mb-6">
            To run <strong>{modelName}</strong> locally, you need to install rbee first.
          </p>
          
          <div className="space-y-4 mb-6">
            <div className="border rounded-lg p-4">
              <h3 className="font-bold mb-2">🐧 Arch Linux</h3>
              <code className="block bg-gray-100 p-2 rounded">
                yay -S rbee-keeper
              </code>
            </div>
            
            <div className="border rounded-lg p-4">
              <h3 className="font-bold mb-2">🐧 Debian/Ubuntu</h3>
              <code className="block bg-gray-100 p-2 rounded">
                curl -fsSL https://rbee.dev/install.sh | sh
              </code>
            </div>
            
            <div className="border rounded-lg p-4 opacity-50">
              <h3 className="font-bold mb-2">🍎 macOS (Coming Soon)</h3>
              <p className="text-sm">Sign up to get notified</p>
              <input 
                type="email" 
                placeholder="your@email.com"
                className="mt-2 px-3 py-2 border rounded w-full"
              />
            </div>
            
            <div className="border rounded-lg p-4 opacity-50">
              <h3 className="font-bold mb-2">🪟 Windows (Coming Soon)</h3>
              <p className="text-sm">Sign up to get notified</p>
              <input 
                type="email" 
                placeholder="your@email.com"
                className="mt-2 px-3 py-2 border rounded w-full"
              />
            </div>
          </div>
          
          <div className="flex gap-4">
            <a 
              href={rbeeUrl}
              className="flex-1 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-center"
            >
              ← Back to {modelName}
            </a>
            <button 
              onClick={onClose}
              className="px-6 py-3 border rounded-lg hover:bg-gray-50"
            >
              Close
            </button>
          </div>
          
          <p className="text-sm text-gray-600 mt-4 text-center">
            After installing, come back and click the button again!
          </p>
        </div>
      </div>
    )
  }
  ```
- [ ] Test modal appears
- [ ] Test close functionality

---

## 👷 Phase 5: Workers Page (Day 5)

### 5.1 Create Workers List Page

- [ ] Create `app/workers/page.tsx`:
  ```tsx
  import { WorkerCatalogClient } from '@rbee/marketplace-sdk'
  import { WorkerCard, MarketplaceGrid } from '@rbee/marketplace-components'
  
  export const metadata = {
    title: 'Worker Binaries - rbee Marketplace',
    description: 'Install optimized worker binaries for your hardware'
  }
  
  export default async function WorkersPage() {
    const client = new WorkerCatalogClient('https://catalog.rbee.dev')
    const workers = await client.listWorkers()
    
    return (
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold mb-8">Worker Binaries</h1>
        
        <MarketplaceGrid
          items={workers}
          renderItem={(worker) => (
            <WorkerCard
              key={worker.id}
              worker={worker}
              installButton={
                <a 
                  href={`rbee://install/worker/${worker.id}`}
                  className="btn-primary"
                >
                  Install Worker
                </a>
              }
              mode="nextjs"
            />
          )}
        />
      </div>
    )
  }
  
  export const revalidate = 3600
  ```
- [ ] Test page renders
- [ ] Test workers appear

### 5.2 Create Worker Detail Page

- [ ] Create `app/workers/[id]/page.tsx` (similar to model detail)
- [ ] Generate static params
- [ ] Generate metadata
- [ ] Add installation button
- [ ] Test page renders

---

## 🚀 Phase 6: Deployment (Day 6)

### 6.1 Configure Cloudflare Pages

- [ ] Create `wrangler.toml`:
  ```toml
  name = "marketplace-rbee-dev"
  compatibility_date = "2024-01-01"
  pages_build_output_dir = ".next"
  ```
- [ ] Build site: `pnpm build`
- [ ] Test static export works
- [ ] Verify all pages generated

### 6.2 Deploy to Cloudflare

- [ ] Install Wrangler: `pnpm add -D wrangler`
- [ ] Login: `wrangler login`
- [ ] Deploy: `pnpm deploy`
- [ ] Verify site is live
- [ ] Test on marketplace.rbee.dev

### 6.3 Configure DNS

- [ ] Add CNAME record: `marketplace.rbee.dev` → Cloudflare Pages
- [ ] Wait for DNS propagation
- [ ] Test site loads on custom domain
- [ ] Verify HTTPS works

### 6.4 Test Production

- [ ] Test home page loads
- [ ] Test models page loads
- [ ] Test model detail pages load
- [ ] Test workers page loads
- [ ] Test "Run with rbee" button
- [ ] Test install modal
- [ ] Test on mobile
- [ ] Test on different browsers

---

## 📊 Phase 7: SEO & Analytics (Day 7)

### 7.1 Generate Sitemap

- [ ] Create `app/sitemap.ts`:
  ```typescript
  import { HuggingFaceClient, WorkerCatalogClient } from '@rbee/marketplace-sdk'
  
  export default async function sitemap() {
    const hfClient = new HuggingFaceClient()
    const workerClient = new WorkerCatalogClient()
    
    const models = await hfClient.listModels({ limit: 1000 })
    const workers = await workerClient.listWorkers()
    
    return [
      {
        url: 'https://marketplace.rbee.dev',
        lastModified: new Date(),
        changeFrequency: 'daily',
        priority: 1
      },
      {
        url: 'https://marketplace.rbee.dev/models',
        lastModified: new Date(),
        changeFrequency: 'daily',
        priority: 0.9
      },
      ...models.map(model => ({
        url: `https://marketplace.rbee.dev/models/${encodeURIComponent(model.id)}`,
        lastModified: new Date(),
        changeFrequency: 'weekly',
        priority: 0.8
      })),
      {
        url: 'https://marketplace.rbee.dev/workers',
        lastModified: new Date(),
        changeFrequency: 'monthly',
        priority: 0.7
      },
      ...workers.map(worker => ({
        url: `https://marketplace.rbee.dev/workers/${worker.id}`,
        lastModified: new Date(),
        changeFrequency: 'monthly',
        priority: 0.6
      }))
    ]
  }
  ```
- [ ] Build and verify sitemap.xml generated
- [ ] Test sitemap is accessible

### 7.2 Add robots.txt

- [ ] Create `app/robots.ts`:
  ```typescript
  export default function robots() {
    return {
      rules: {
        userAgent: '*',
        allow: '/',
      },
      sitemap: 'https://marketplace.rbee.dev/sitemap.xml',
    }
  }
  ```
- [ ] Verify robots.txt generated

### 7.3 Submit to Search Engines

- [ ] Submit sitemap to Google Search Console
- [ ] Submit sitemap to Bing Webmaster Tools
- [ ] Verify indexing starts
- [ ] Monitor search rankings

### 7.4 Add Analytics (Optional)

- [ ] Add Plausible Analytics script
- [ ] Track page views
- [ ] Track button clicks
- [ ] Monitor conversion rate

---

## ✅ Success Criteria

### Must Have

- [ ] Site deployed to marketplace.rbee.dev
- [ ] 1000+ model pages pre-rendered
- [ ] All worker pages pre-rendered
- [ ] "Run with rbee" button works
- [ ] Installation detection works
- [ ] Install modal works
- [ ] Sitemap generated
- [ ] robots.txt present
- [ ] Mobile responsive
- [ ] Fast loading (<3s)

### Nice to Have

- [ ] Search functionality
- [ ] Filter functionality
- [ ] Analytics tracking
- [ ] Error tracking
- [ ] A/B testing

---

## 🚀 Deliverables

1. **Site:** marketplace.rbee.dev live and accessible
2. **Pages:** 1000+ model pages + worker pages
3. **SEO:** Sitemap, robots.txt, metadata
4. **Integration:** Protocol detection and fallback
5. **Performance:** Fast loading, optimized images

---

## 📝 Notes

### Key Principles

1. **SSG EVERYTHING** - Pre-render all pages at build time
2. **SEO OPTIMIZED** - Metadata, sitemap, robots.txt
3. **INSTALLATION AWARE** - Detect rbee, fallback gracefully
4. **MOBILE FIRST** - Responsive design
5. **FAST** - Optimize images, minimize JS

### Common Pitfalls

- ❌ Don't use client-side rendering for main content
- ❌ Don't forget metadata for each page
- ❌ Don't hardcode URLs (use env vars)
- ✅ Pre-render everything possible
- ✅ Add proper error handling
- ✅ Test on real devices

---

**Complete each phase, test thoroughly, deploy with confidence!** ✅
