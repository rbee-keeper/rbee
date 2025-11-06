// TEAM-415: Pure SSG page for maximum SEO
// TEAM-421: Show top 100 popular models (WASM filtering doesn't work in SSG)
import { listHuggingFaceModels } from '@rbee/marketplace-node'
import type { ModelTableItem } from '@rbee/ui/marketplace'
import { ModelTableWithRouting } from '@/components/ModelTableWithRouting'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'AI Language Models | Marketplace',
  description: 'Browse compatible AI language models from HuggingFace. Pre-rendered for instant loading and maximum SEO.',
}

export default async function ModelsPage() {
  // TEAM-421: WASM doesn't work in Next.js SSG - show top 100 popular models
  // TODO: Add client-side compatibility filtering in the future
  const FETCH_LIMIT = 100
  
  console.log(`[SSG] Fetching top ${FETCH_LIMIT} most popular models`)
  
  const hfModels = await listHuggingFaceModels({ limit: FETCH_LIMIT })
  
  console.log(`[SSG] Showing ${hfModels.length} models`)
  
  const models: ModelTableItem[] = hfModels.map((model) => {
    const m = model as unknown as Record<string, unknown>
    return {
      id: m.id as string,
      name: (m.name as string) || (m.id as string),
      description: (m.description as string) || `${(m.author as string) || 'Community'} model`,
      author: m.author as string | undefined,
      downloads: (m.downloads as number) ?? 0,
      likes: (m.likes as number) ?? 0,
      tags: (m.tags as string[] | undefined)?.slice(0, 10) ?? []
    }
  })

  return (
    <div className="container mx-auto px-4 py-12 max-w-7xl">
      {/* Header Section */}
      <div className="mb-12 space-y-4">
        <div className="space-y-2">
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
            LLM Models
          </h1>
          <p className="text-muted-foreground text-lg md:text-xl max-w-3xl">
            Discover and download state-of-the-art language models from HuggingFace
          </p>
        </div>
        
        {/* Stats */}
        <div className="flex items-center gap-6 text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <div className="size-2 rounded-full bg-primary" />
            <span>{models.length.toLocaleString()} models available</span>
          </div>
        </div>
      </div>

      {/* Table with client-side routing (minimal JS for navigation only) */}
      <div className="rounded-lg border border-border bg-card p-6">
        <ModelTableWithRouting models={models} />
      </div>
    </div>
  )
}
