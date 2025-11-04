# Checklist 03: Next.js Marketplace Site

**Timeline:** 1 week  
**Status:** 📋 NOT STARTED  
**Dependencies:** Checklist 01 (Components), Checklist 02 (SDK)  
**TEAM-400:** ✅ RULE ZERO - App EXISTS, just add content

---

## 🎯 Goal

Update EXISTING `frontend/apps/marketplace/` to use marketplace components and SDK. Pre-render 1000+ model pages with SSG. Deploy to Cloudflare Pages (already configured).

**TEAM-400:** App is already set up! Just add pages + data fetching.

---

## 📦 Phase 1: Setup Dependencies (Day 1, Morning)

**TEAM-400:** App exists at `frontend/apps/marketplace/` with Next.js 15 + Cloudflare configured.

### 1.1 Add Workspace Packages

- [ ] Navigate to: `cd frontend/apps/marketplace/`
- [ ] Update `package.json` dependencies:
  ```json
  {
    "dependencies": {
      "@opennextjs/cloudflare": "^1.3.0",
      "@rbee/ui": "workspace:*",
      "@rbee/marketplace-sdk": "workspace:*",
      "next": "15.4.6",
      "react": "19.1.0",
      "react-dom": "19.1.0"
    }
  }
  ```
- [ ] Install: `pnpm install`
- [ ] Verify dev server: `pnpm dev`

### 1.2 Configure Tailwind (if needed)

- [ ] Check if `tailwind.config.ts` extends `@repo/tailwind-config`
- [ ] If not, update to use workspace config:
  ```typescript
  import type { Config } from 'tailwindcss'
  import sharedConfig from '@repo/tailwind-config'
  
  const config: Config = {
    ...sharedConfig,
    content: [
      './app/**/*.{js,ts,jsx,tsx,mdx}',
      './components/**/*.{js,ts,jsx,tsx,mdx}',
      // TEAM-400: Include rbee-ui components
      '../../packages/rbee-ui/src/**/*.{js,ts,jsx,tsx}',
    ],
  }
  export default config
  ```

---

## 🏠 Phase 2: Home Page (Day 1, Afternoon)

**TEAM-400:** Replace default Next.js content with marketplace home.

### 2.1 Update Home Page

- [ ] Replace `app/page.tsx`:
  ```tsx
  // TEAM-400: Marketplace home page
  import { ModelsPage, defaultModelsPageProps } from '@rbee/ui/marketplace/pages/ModelsPage'
  
  export default function Home() {
    return <ModelsPage {...defaultModelsPageProps} />
  }
  ```
- [ ] Update `app/layout.tsx` metadata:
  ```tsx
  export const metadata: Metadata = {
    title: 'rbee Marketplace - Browse AI Models & Workers',
    description: 'Discover and download AI models from HuggingFace and CivitAI. Find workers for your rbee cluster.',
  }
  ```
- [ ] Test: `pnpm dev` and visit `http://localhost:3000`
- [ ] Verify ModelsPage renders correctly

### 2.2 Add Navigation (Optional)

- [ ] Create `app/components/Nav.tsx`:
  ```tsx
  // TEAM-400: Simple marketplace navigation
  import Link from 'next/link'
  
  export function Nav() {
    return (
      <nav className="border-b">
        <div className="container mx-auto px-4 py-4 flex gap-6">
          <Link href="/" className="font-bold">rbee Marketplace</Link>
          <Link href="/models">Models</Link>
          <Link href="/workers">Workers</Link>
        </div>
      </nav>
    )
  }
  ```
- [ ] Add to `app/layout.tsx`:
  ```tsx
  import { Nav } from './components/Nav'
  
  export default function RootLayout({ children }: { children: React.ReactNode }) {
    return (
      <html lang="en">
        <body>
          <Nav />
          {children}
        </body>
      </html>
    )
  }
  ```

---

## 📄 Phase 3: Models Pages (Days 2-3)

**TEAM-400:** Create models list + detail pages with SSG.

### 3.1 Models List Page

- [ ] Create `app/models/page.tsx`:
  ```tsx
  // TEAM-400: Models list page with SSG
  import { ModelsPage } from '@rbee/ui/marketplace/pages/ModelsPage'
  import type { ModelsPageProps } from '@rbee/ui/marketplace/pages/ModelsPage'
  import { HuggingFaceClient } from '@rbee/marketplace-sdk'
  
  // TEAM-400: Fetch data at build time (SSG)
  async function getModelsData(): Promise<ModelsPageProps> {
    const client = new HuggingFaceClient()
    const models = await client.list_models({ limit: 100 })
    
    return {
      seo: {
        title: 'AI Models | rbee Marketplace',
        description: 'Browse AI models from HuggingFace'
      },
      template: {
        title: 'AI Models',
        description: 'Discover and download AI models',
        models: models.map(m => ({
          model: m,
          onAction: undefined, // Client-side only
        })),
        filters: { search: '', sort: 'popular' }
      }
    }
  }
  
  export default async function ModelsListPage() {
    const props = await getModelsData()
    return <ModelsPage {...props} />
  }
  
  // TEAM-400: Generate metadata for SEO
  export async function generateMetadata() {
    return {
      title: 'AI Models | rbee Marketplace',
      description: 'Browse and download AI models from HuggingFace and CivitAI'
    }
  }
  ```

### 3.2 Model Detail Page (Dynamic Route)

- [ ] Create `app/models/[modelId]/page.tsx`:
  ```tsx
  // TEAM-400: Model detail page with SSG
  import { ModelDetailPage } from '@rbee/ui/marketplace/pages/ModelDetailPage'
  import type { ModelDetailPageProps } from '@rbee/ui/marketplace/pages/ModelDetailPage'
  import { HuggingFaceClient } from '@rbee/marketplace-sdk'
  
  interface PageProps {
    params: { modelId: string }
  }
  
  // TEAM-400: Fetch single model data
  async function getModelData(modelId: string): Promise<ModelDetailPageProps> {
    const client = new HuggingFaceClient()
    const model = await client.get_model(modelId)
    
    return {
      seo: {
        title: `${model.name} | rbee Marketplace`,
        description: model.description
      },
      template: {
        model,
        installButton: undefined, // Client-side component
      }
    }
  }
  
  export default async function ModelDetailPageRoute({ params }: PageProps) {
    const props = await getModelData(params.modelId)
    return <ModelDetailPage {...props} />
  }
  
  // TEAM-400: Generate static params for top 1000 models
  export async function generateStaticParams() {
    const client = new HuggingFaceClient()
    const models = await client.list_models({ limit: 1000, sort: 'popular' })
    
    return models.map((model) => ({
      modelId: model.id,
    }))
  }
  
  // TEAM-400: Generate metadata for each model
  export async function generateMetadata({ params }: PageProps) {
    const props = await getModelData(params.modelId)
    return {
      title: props.seo.title,
      description: props.seo.description,
    }
  }
  ```

### 3.3 Test SSG Build

- [ ] Build site: `pnpm build`
- [ ] Check output:
  ```
  ✓ Generating static pages (1000/1000)
  ```
- [ ] Verify pages generated: `ls .next/server/app/models/`
- [ ] Test production: `pnpm start`

---

## 👷 Phase 4: Workers Pages (Day 4)

**TEAM-400:** Similar to models, but for workers.

### 4.1 Workers List Page

- [ ] Create `app/workers/page.tsx`:
  ```tsx
  // TEAM-400: Workers list page
  import { WorkersPage } from '@rbee/ui/marketplace/pages/WorkersPage'
  import { WorkerClient } from '@rbee/marketplace-sdk'
  
  async function getWorkersData() {
    // TEAM-400: This talks to rbee-hive
    // For SSG, we need a public endpoint or mock data
    const client = new WorkerClient('https://api.rbee.dev')
    const workers = await client.list_workers()
    
    return {
      seo: {
        title: 'Workers | rbee Marketplace',
        description: 'Browse rbee workers'
      },
      template: {
        title: 'Workers',
        workers: workers.map(w => ({
          worker: w,
          onAction: undefined,
        })),
      }
    }
  }
  
  export default async function WorkersListPage() {
    const props = await getWorkersData()
    return <WorkersPage {...props} />
  }
  ```

### 4.2 Worker Detail Page

- [ ] Create `app/workers/[workerId]/page.tsx`:
  ```tsx
  // TEAM-400: Worker detail page
  import { WorkerDetailPage } from '@rbee/ui/marketplace/pages/WorkerDetailPage'
  import { WorkerClient } from '@rbee/marketplace-sdk'
  
  interface PageProps {
    params: { workerId: string }
  }
  
  async function getWorkerData(workerId: string) {
    const client = new WorkerClient('https://api.rbee.dev')
    const worker = await client.get_worker(workerId)
    
    return {
      seo: {
        title: `${worker.name} | rbee Marketplace`,
        description: worker.description
      },
      template: { worker }
    }
  }
  
  export default async function WorkerDetailPageRoute({ params }: PageProps) {
    const props = await getWorkerData(params.workerId)
    return <WorkerDetailPage {...props} />
  }
  
  export async function generateStaticParams() {
    const client = new WorkerClient('https://api.rbee.dev')
    const workers = await client.list_workers()
    
    return workers.map((worker) => ({
      workerId: worker.id,
    }))
  }
  ```

---

## 🔘 Phase 5: Installation Detection (Day 5)

**TEAM-400:** Client-side detection if Keeper is installed.

### 5.1 Create Detection Hook

- [ ] Create `app/hooks/useKeeperInstalled.ts`:
  ```tsx
  'use client'
  
  import { useEffect, useState } from 'react'
  
  export function useKeeperInstalled() {
    const [installed, setInstalled] = useState(false)
    const [checking, setChecking] = useState(true)
    
    useEffect(() => {
      // TEAM-400: Try to detect rbee:// protocol support
      async function checkInstallation() {
        try {
          // Method 1: Try to open rbee:// URL
          const testUrl = 'rbee://ping'
          const iframe = document.createElement('iframe')
          iframe.style.display = 'none'
          iframe.src = testUrl
          document.body.appendChild(iframe)
          
          // If no error after 1 second, assume installed
          await new Promise(resolve => setTimeout(resolve, 1000))
          setInstalled(true)
          
          document.body.removeChild(iframe)
        } catch (error) {
          setInstalled(false)
        } finally {
          setChecking(false)
        }
      }
      
      checkInstallation()
    }, [])
    
    return { installed, checking }
  }
  ```

### 5.2 Create Install Button Component

- [ ] Create `app/components/InstallButton.tsx`:
  ```tsx
  'use client'
  
  import { useKeeperInstalled } from '../hooks/useKeeperInstalled'
  import { Button } from '@rbee/ui/atoms/Button'
  
  interface InstallButtonProps {
    modelId: string
  }
  
  export function InstallButton({ modelId }: InstallButtonProps) {
    const { installed, checking } = useKeeperInstalled()
    
    if (checking) {
      return <Button disabled>Checking...</Button>
    }
    
    if (installed) {
      // TEAM-400: Open rbee:// protocol
      return (
        <Button onClick={() => {
          window.location.href = `rbee://model/${modelId}`
        }}>
          Run with rbee
        </Button>
      )
    }
    
    // TEAM-400: Download Keeper
    return (
      <Button onClick={() => {
        window.location.href = 'https://github.com/veighnsche/llama-orch/releases'
      }}>
        Download Keeper
      </Button>
    )
  }
  ```

### 5.3 Use in Model Detail Page

- [ ] Update `app/models/[modelId]/page.tsx`:
  ```tsx
  import { InstallButton } from '@/components/InstallButton'
  
  // In getModelData:
  return {
    // ...
    template: {
      model,
      installButton: <InstallButton modelId={model.id} />,
    }
  }
  ```

---

## 🌐 Phase 6: SEO & Sitemap (Day 6)

### 6.1 Generate Sitemap

- [ ] Create `app/sitemap.ts`:
  ```tsx
  // TEAM-400: Generate sitemap for all models + workers
  import { MetadataRoute } from 'next'
  import { HuggingFaceClient, WorkerClient } from '@rbee/marketplace-sdk'
  
  export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
    const baseUrl = 'https://marketplace.rbee.dev'
    
    // Fetch all models
    const hfClient = new HuggingFaceClient()
    const models = await hfClient.list_models({ limit: 1000 })
    
    // Fetch all workers
    const workerClient = new WorkerClient('https://api.rbee.dev')
    const workers = await workerClient.list_workers()
    
    return [
      {
        url: baseUrl,
        lastModified: new Date(),
        changeFrequency: 'daily',
        priority: 1,
      },
      {
        url: `${baseUrl}/models`,
        lastModified: new Date(),
        changeFrequency: 'daily',
        priority: 0.9,
      },
      ...models.map((model) => ({
        url: `${baseUrl}/models/${model.id}`,
        lastModified: new Date(),
        changeFrequency: 'weekly' as const,
        priority: 0.8,
      })),
      {
        url: `${baseUrl}/workers`,
        lastModified: new Date(),
        changeFrequency: 'weekly',
        priority: 0.7,
      },
      ...workers.map((worker) => ({
        url: `${baseUrl}/workers/${worker.id}`,
        lastModified: new Date(),
        changeFrequency: 'monthly' as const,
        priority: 0.6,
      })),
    ]
  }
  ```

### 6.2 Add robots.txt

- [ ] Create `app/robots.ts`:
  ```tsx
  import { MetadataRoute } from 'next'
  
  export default function robots(): MetadataRoute.Robots {
    return {
      rules: {
        userAgent: '*',
        allow: '/',
      },
      sitemap: 'https://marketplace.rbee.dev/sitemap.xml',
    }
  }
  ```

### 6.3 Add Open Graph Images

- [ ] Create `app/opengraph-image.tsx`:
  ```tsx
  import { ImageResponse } from 'next/og'
  
  export const size = {
    width: 1200,
    height: 630,
  }
  
  export default async function Image() {
    return new ImageResponse(
      (
        <div style={{
          fontSize: 128,
          background: 'white',
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}>
          rbee Marketplace
        </div>
      ),
      { ...size }
    )
  }
  ```

---

## 🚀 Phase 7: Deploy to Cloudflare Pages (Day 7)

**TEAM-400:** Already configured! Just deploy.

### 7.1 Build for Production

- [ ] Build: `pnpm build`
- [ ] Check output size:
  ```bash
  du -sh .next
  ```
- [ ] Verify all pages generated:
  ```bash
  find .next/server/app -name "*.html" | wc -l
  ```

### 7.2 Test Cloudflare Build

- [ ] Build with OpenNext: `pnpm run deploy`
- [ ] Or preview: `pnpm run preview`
- [ ] Test locally with Wrangler

### 7.3 Deploy

- [ ] Deploy to Cloudflare Pages:
  ```bash
  pnpm run deploy
  ```
- [ ] Or connect to GitHub for auto-deploy
- [ ] Verify deployment: `https://marketplace.rbee.dev`

### 7.4 Configure Custom Domain

- [ ] In Cloudflare Pages dashboard
- [ ] Add custom domain: `marketplace.rbee.dev`
- [ ] Configure DNS
- [ ] Enable HTTPS

---

## ✅ Success Criteria

### Must Have

- [ ] Home page renders with ModelsPage
- [ ] Models list page works (/models)
- [ ] Model detail pages work (/models/[id])
- [ ] Workers list page works (/workers)
- [ ] Worker detail pages work (/workers/[id])
- [ ] SSG generates 1000+ model pages
- [ ] Installation detection works
- [ ] "Run with rbee" button works (if Keeper installed)
- [ ] "Download Keeper" button works (if not installed)
- [ ] Sitemap generated
- [ ] Deployed to Cloudflare Pages
- [ ] Custom domain works

### Nice to Have

- [ ] Search functionality
- [ ] Filter by category
- [ ] Sort options
- [ ] Related models
- [ ] Download stats
- [ ] User reviews
- [ ] Analytics (Cloudflare Analytics)

---

## 🚀 Deliverables

1. **Next.js App:** Updated `frontend/apps/marketplace/`
2. **Pages:** Home, Models List, Model Detail, Workers List, Worker Detail
3. **SSG:** 1000+ pre-rendered model pages
4. **SEO:** Sitemap, robots.txt, Open Graph images
5. **Installation Detection:** Client-side Keeper detection
6. **Deployment:** Live on Cloudflare Pages

---

## 📝 Notes

### Key Principles

1. **APP EXISTS** - Don't create from scratch, update existing
2. **SSG FIRST** - Pre-render everything at build time
3. **CLIENT COMPONENTS** - Only for interactive parts (buttons, detection)
4. **USE rbee-ui** - Import marketplace components
5. **USE marketplace-sdk** - WASM SDK for data fetching

### Common Pitfalls

- ❌ Don't create new Next.js app (it exists!)
- ❌ Don't fetch data client-side (use SSG)
- ❌ Don't hardcode model data (fetch from SDK)
- ❌ Don't forget generateStaticParams (needed for SSG)
- ✅ Use existing app structure
- ✅ Fetch data at build time
- ✅ Use marketplace-sdk (WASM)
- ✅ Generate static params for all models

### SSG vs Client-Side

**SSG (Build Time):**
- Model list
- Model details
- Worker list
- Worker details
- SEO metadata

**Client-Side (Runtime):**
- Installation detection
- "Run with rbee" button
- Search/filter (optional)
- Real-time data (optional)

---

**Start with Phase 1, use existing app!** ✅

**TEAM-400 🐝🎊**
