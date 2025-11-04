// TEAM-405: SSG page with client wrapper for interactivity
import { fetchTopModels } from '@/lib/huggingface'
import type { ModelTableItem } from '@rbee/ui/marketplace'
import { ModelListClient } from '@/components/ModelListClient'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'AI Language Models | Marketplace',
  description: 'Browse top AI language models',
}

export default async function ModelsPage() {
  // SSG: Fetch at build time
  const hfModels = await fetchTopModels(100)
  
  const models: ModelTableItem[] = hfModels.map((model: any) => ({
    id: model.id,
    name: model.id.split('/').pop() || model.id,
    description: `${model.author || 'Community'} - ${model.pipeline_tag || 'text-generation'}`,
    author: model.author || null,
    downloads: model.downloads,
    likes: model.likes,
    tags: model.tags.slice(0, 10)
  }))

  return (
    <main className="container mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">AI Language Models</h1>
        <p className="text-muted-foreground text-lg">
          Discover {models.length}+ state-of-the-art language models
        </p>
      </div>

      {/* Client component for interactivity */}
      <ModelListClient initialModels={models} />
    </main>
  )
}
