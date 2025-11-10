// TEAM-460: HuggingFace model detail page (migrated from /models/[slug])
// TEAM-415: SSG model detail page with slugified URLs
// TEAM-410: Added compatibility integration
// TEAM-413: Added InstallButton integration
// TEAM-421: Pre-build top 100 popular models (WASM filtering doesn't work in SSG)
// TEAM-464: Updated to use getRawHuggingFaceModel for complete HF data
// TEAM-464: Added README.md fetching at build time (passed as markdown to react-markdown)
import { getRawHuggingFaceModel, getHuggingFaceModelReadme, listHuggingFaceModels } from '@rbee/marketplace-node'
import type { Metadata } from 'next'
import { notFound } from 'next/navigation'
import { ModelDetailWithInstall } from '@/components/ModelDetailWithInstall'
import { modelIdToSlug, slugToModelId } from '@/lib/slugify'

interface Props {
  params: Promise<{ slug: string }>
}

export async function generateStaticParams() {
  // TEAM-421: WASM doesn't work in Next.js SSG build context
  // Solution: Pre-build top 100 models, filter client-side at runtime
  const FETCH_LIMIT = 100

  console.log(
    `[SSG] Pre-building top ${FETCH_LIMIT} most popular HuggingFace models (compatibility check happens at runtime)`,
  )

  const models = await listHuggingFaceModels({ limit: FETCH_LIMIT })

  console.log(`[SSG] Pre-building ${models.length} HuggingFace model pages`)

  return models.map((model) => ({
    slug: modelIdToSlug((model as { id: string }).id),
  }))
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params
  const modelId = slugToModelId(slug)

  try {
    // TEAM-464: Use raw HF model for metadata
    const model = await getRawHuggingFaceModel(modelId)
    const parts = model.id.split('/')
    const name = parts.length >= 2 ? parts[1] : model.id

    return {
      title: `${name} | HuggingFace Model`,
      description: model.cardData?.model_description || model.description || `${name} - ${model.downloads?.toLocaleString()} downloads`,
    }
  } catch {
    return { title: 'Model Not Found' }
  }
}

export default async function HuggingFaceModelPage({ params }: Props) {
  const { slug } = await params
  const modelId = slugToModelId(slug)

  try {
    // TEAM-464: Get raw HuggingFace model with ALL fields
    const hfModel = await getRawHuggingFaceModel(modelId)
    
    // TEAM-464: Fetch README.md at build time (SSG)
    // Pass as raw markdown to MarkdownContent component (uses react-markdown)
    let readmeMarkdown: string | undefined
    try {
      readmeMarkdown = await getHuggingFaceModelReadme(modelId) || undefined
    } catch (readmeError) {
      // README fetch failed - not critical, continue without it
      console.warn(`[SSG] Failed to fetch README for ${modelId}:`, readmeError)
    }
    
    // Convert to display format
    const parts = hfModel.id.split('/')
    const name = parts.length >= 2 ? parts[1] : hfModel.id
    const author = parts.length >= 2 ? parts[0] : hfModel.author
    
    // Calculate total size
    const totalBytes = hfModel.siblings?.reduce((sum, file) => sum + (file.size || 0), 0) || 0
    const formatBytes = (bytes: number): string => {
      if (bytes === 0) return '0 B'
      const k = 1024
      const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
      const i = Math.floor(Math.log(bytes) / Math.log(k))
      return `${Math.round((bytes / k ** i) * 100) / 100} ${sizes[i]}`
    }
    
    const model = {
      id: hfModel.id,
      name,
      author,
      description: hfModel.cardData?.model_description || hfModel.description || '',
      downloads: hfModel.downloads || 0,
      likes: hfModel.likes || 0,
      size: formatBytes(totalBytes),
      tags: hfModel.tags || [],
      pipeline_tag: hfModel.pipeline_tag,
      library_name: hfModel.library_name,
      sha: hfModel.sha,
      mask_token: hfModel.mask_token,
      widgetData: hfModel.widgetData,
      config: hfModel.config,
      cardData: hfModel.cardData,
      transformersInfo: hfModel.transformersInfo,
      inference: hfModel.inference,
      safetensors: hfModel.safetensors,
      spaces: hfModel.spaces,
      siblings: hfModel.siblings?.map(s => ({ filename: s.rfilename, size: s.size || 0 })),
      createdAt: hfModel.createdAt,
      lastModified: hfModel.lastModified,
      readmeMarkdown, // TEAM-464: Raw markdown for react-markdown
    }

    // TEAM-410: Get compatible workers (build time)
    // Note: In a real implementation, this would call marketplace-node's
    // checkModelCompatibility function for each worker type.
    // For now, we'll pass undefined and let the component handle it gracefully.
    const compatibleWorkers = undefined

    return (
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <ModelDetailWithInstall model={model} compatibleWorkers={compatibleWorkers} />
      </div>
    )
  } catch {
    notFound()
  }
}
