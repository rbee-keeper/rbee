// TEAM-405: Pure SSG page for maximum SEO
import { fetchTopModels } from '@/lib/huggingface'
import type { ModelTableItem } from '@rbee/ui/marketplace'
import { ModelTableWithRouting } from '@/components/ModelTableWithRouting'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'AI Language Models | Marketplace',
  description: 'Browse top AI language models from HuggingFace. Pre-rendered for instant loading and maximum SEO.',
}

export default async function ModelsPage() {
  // SSG: Fetch at build time - pure static HTML
  const hfModels = await fetchTopModels(100) as any[]
  
  const models: ModelTableItem[] = hfModels.map((model) => ({
    id: model.id,
    name: model.id.split('/').pop() || model.id,
    description: `${model.author || 'Community'} - ${model.pipeline_tag || 'text-generation'}`,
    author: model.author || null,
    downloads: model.downloads,
    likes: model.likes,
    tags: model.tags.slice(0, 10)
  }))

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
