// TEAM-476: Marketplace homepage - simple and static
// TEAM-481: Enhanced with SSR trending models (CivitAI top 10, HuggingFace top 9)

import type { CivitAIListModelsParams, HuggingFaceListModelsParams } from '@rbee/marketplace-core'
import { Button } from '@rbee/ui/atoms'
import { HFModelListCard, ModelCardVertical } from '@rbee/ui/marketplace'
import { ArrowRight } from 'lucide-react'
import Link from 'next/link'
import { fetchModels } from '@/lib/fetchModels'

export default async function Home() {
  // TEAM-481: Fetch trending models server-side for SEO
  const [civitaiResponse, huggingfaceResponse] = await Promise.all([
    fetchModels('civitai', {
      sort: 'Most Downloaded',
      limit: 9,
    } as CivitAIListModelsParams),
    fetchModels('huggingface', {
      sort: 'downloads',
      limit: 9,
    } as HuggingFaceListModelsParams),
  ])

  const trendingCivitAI = civitaiResponse.items
  const trendingHuggingFace = huggingfaceResponse.items

  return (
    <div className="container mx-auto px-4 py-16">
      <div className="max-w-7xl mx-auto space-y-16">
        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold">rbee Marketplace</h1>
          <p className="text-xl text-muted-foreground">Browse AI models and workers for your rbee cluster</p>
        </div>

        {/* Trending HuggingFace Models - 3×3 grid */}
        <section className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-3xl font-bold">Trending LLM Models</h2>
              <p className="text-muted-foreground mt-1">Top 9 most downloaded from HuggingFace</p>
            </div>
            <Button asChild variant="outline">
              <Link href="/models/huggingface" className="flex items-center gap-2">
                View All
                <ArrowRight className="size-4" />
              </Link>
            </Button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {trendingHuggingFace.map((model) => (
              <HFModelListCard
                key={model.id}
                href={`/models/huggingface/${encodeURIComponent(model.id)}`}
                model={{
                  id: model.id,
                  name: model.name,
                  author: model.author,
                  type: model.type,
                  downloads: model.downloads,
                  likes: model.likes,
                }}
              />
            ))}
          </div>
        </section>

        {/* Trending CivitAI Models - 2 rows × 5 cols */}
        <section className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-3xl font-bold">Trending Image Models</h2>
              <p className="text-muted-foreground mt-1">Top 9 most downloaded from CivitAI</p>
            </div>
            <Button asChild variant="outline">
              <Link href="/models/civitai" className="flex items-center gap-2">
                View All
                <ArrowRight className="size-4" />
              </Link>
            </Button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-6">
            {trendingCivitAI.map((model) => (
              <ModelCardVertical
                key={model.id}
                href={`/models/civitai/${model.id}`}
                model={{
                  id: model.id,
                  name: model.name,
                  description: model.description || '',
                  ...(model.author ? { author: model.author } : {}),
                  ...(model.imageUrl ? { imageUrl: model.imageUrl } : {}),
                  tags: model.tags.slice(0, 3),
                  downloads: model.downloads,
                  likes: model.likes,
                  size: model.sizeBytes ? `${(model.sizeBytes / (1024 * 1024 * 1024)).toFixed(2)} GB` : model.type,
                }}
              />
            ))}
          </div>
        </section>
      </div>
    </div>
  )
}
