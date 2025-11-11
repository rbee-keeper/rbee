// TEAM-460: Civitai model detail page with slugified URLs
// TEAM-463: Updated to use CivitAI-style layout
// TEAM-464: Using manifest-based SSG (Phase 2)
import { getCivitaiModel } from '@rbee/marketplace-node'
import { CivitAIModelDetail } from '@rbee/ui/marketplace'
import type { Metadata } from 'next'
import { notFound, redirect } from 'next/navigation'
import { InstallCTA } from '@/components/InstallCTA'
import { loadModelsBySource } from '@/lib/manifests'
import { slugToModelId } from '@/lib/slugify'

interface Props {
  params: Promise<{ slug: string }>
}

export async function generateStaticParams() {
  console.log('[SSG] Generating CivitAI model pages from manifest')

  // TEAM-464: Read from manifest instead of API
  const models = await loadModelsBySource('civitai')

  console.log(`[SSG] Pre-building ${models.length} CivitAI model pages`)

  return models.map((model) => ({
    slug: model.slug,
  }))
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params

  // TEAM-460: Check if this is actually a filter keyword (route conflict fix)
  const oldFilterKeywords = ['month', 'week', 'checkpoints', 'loras', 'sdxl', 'sd15']
  if (oldFilterKeywords.includes(slug)) {
    return { title: 'CivitAI Models' }
  }

  const modelId = slugToModelId(slug)

  // Extract Civitai ID from "civitai-{id}" format
  const civitaiId = parseInt(modelId.replace('civitai-', ''), 10)

  if (Number.isNaN(civitaiId)) {
    return { title: 'Model Not Found' }
  }

  try {
    const model = await getCivitaiModel(civitaiId)

    // TEAM-422: Handle optional fields safely
    const downloads = model.stats?.downloadCount || 0
    const description =
      model.description?.substring(0, 160) || `${model.name} - ${downloads.toLocaleString()} downloads`

    return {
      title: `${model.name} | Civitai Model`,
      description,
    }
  } catch {
    return { title: 'Model Not Found' }
  }
}

export default async function CivitaiModelPage({ params }: Props) {
  const { slug } = await params

  // TEAM-460: Check if this is actually a filter keyword (route conflict fix)
  // Redirect old filter URLs (e.g., /month) to new structure (e.g., /filter/month)
  const oldFilterKeywords = ['month', 'week', 'checkpoints', 'loras', 'sdxl', 'sd15']
  if (oldFilterKeywords.includes(slug)) {
    redirect(`/models/civitai/filter/${slug}`)
  }

  const modelId = slugToModelId(slug)

  // Extract Civitai ID from "civitai-{id}" format
  const civitaiId = parseInt(modelId.replace('civitai-', ''), 10)

  if (Number.isNaN(civitaiId)) {
    notFound()
  }

  try {
    const civitaiModel = await getCivitaiModel(civitaiId)

    // TEAM-422: Handle optional fields safely
    const latestVersion = civitaiModel.modelVersions?.[0]
    const totalBytes = latestVersion?.files?.reduce((sum: number, file: any) => sum + (file.sizeKb || 0) * 1024, 0) || 0
    const formatBytes = (bytes: number): string => {
      if (bytes === 0) return '0 B'
      const k = 1024
      const sizes = ['B', 'KB', 'MB', 'GB']
      const i = Math.floor(Math.log(bytes) / Math.log(k))
      return `${(bytes / k ** i).toFixed(2)} ${sizes[i]}`
    }

    // TEAM-463: Convert to CivitAI detail format
    const model = {
      id: `civitai-${civitaiModel.id}`,
      name: civitaiModel.name,
      description: civitaiModel.description || '',
      author: civitaiModel.creator?.username || 'Unknown',
      downloads: civitaiModel.stats?.downloadCount || 0,
      likes: civitaiModel.stats?.favoriteCount || 0,
      rating: civitaiModel.stats?.rating || 0,
      size: formatBytes(totalBytes),
      tags: civitaiModel.tags || [],
      type: civitaiModel.type,
      baseModel: latestVersion?.baseModel || 'Unknown',
      version: latestVersion?.name || 'Latest',
      images: latestVersion?.images?.filter((img) => !img.nsfw).slice(0, 5) || [],
      files: latestVersion?.files || [],
      trainedWords: latestVersion?.trainedWords || [],
      allowCommercialUse: civitaiModel.allowCommercialUse || 'Unknown',
      externalUrl: `https://civitai.com/models/${civitaiModel.id}`,
      externalLabel: 'View on CivitAI',
    }

    return (
      <div className="container mx-auto px-4 py-8 max-w-7xl space-y-6">
        {/* TEAM-463: Conversion CTA */}
        <InstallCTA artifactType="model" artifactName={model.name} />

        {/* TEAM-463: CivitAI-style detail page */}
        <CivitAIModelDetail model={model} />
      </div>
    )
  } catch {
    notFound()
  }
}
