// TEAM-460: HuggingFace model detail page (migrated from /models/[slug])
// TEAM-415: SSG model detail page with slugified URLs
// TEAM-410: Added compatibility integration
// TEAM-413: Added InstallButton integration
// TEAM-421: Pre-build top 100 popular models (WASM filtering doesn't work in SSG)
// TEAM-464: Updated to use getRawHuggingFaceModel for complete HF data
// TEAM-464: Added README.md fetching at build time (passed as markdown to react-markdown)
// TEAM-464: Using manifest-based SSG (Phase 2)
import { getHuggingFaceModel } from '@rbee/marketplace-node'
import type { Metadata } from 'next'
import { notFound } from 'next/navigation'
import { ModelDetailWithInstall } from '@/components/ModelDetailWithInstall'
import { loadModelsBySource, shouldSkipModel } from '@/lib/manifests'
import { slugToModelId } from '@/lib/slugify'

interface Props {
  params: Promise<{ slug: string }>
}

export async function generateStaticParams() {
  console.log('[SSG] Generating HuggingFace model pages from manifest')

  // TEAM-464: Read from manifest instead of API
  const models = await loadModelsBySource('huggingface')

  // Filter out problematic models
  const validModels = models.filter((model) => !shouldSkipModel(model.id))

  const skippedCount = models.length - validModels.length
  if (skippedCount > 0) {
    console.log(`[SSG] Skipping ${skippedCount} problematic models`)
  }

  console.log(`[SSG] Pre-building ${validModels.length} HuggingFace model pages`)

  return validModels.map((model) => ({
    slug: model.slug,
  }))
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params
  const modelId = slugToModelId(slug)

  try {
    // TEAM-464: Use HF model for metadata
    const hfModel = await getHuggingFaceModel(modelId)
    const parts = hfModel.id.split('/')
    const name = parts.length >= 2 ? parts[1] : hfModel.id

    return {
      title: `${name} | HuggingFace Model`,
      description:
        hfModel.cardData?.model_description ||
        hfModel.description ||
        `${name} - ${hfModel.downloads?.toLocaleString()} downloads`,
    }
  } catch {
    return { title: 'Model Not Found' }
  }
}

// TEAM-464: Helper to sanitize nested objects that might cause React rendering errors
function _sanitizeValue(value: unknown): unknown {
  if (value === null || value === undefined) return value
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return value
  if (Array.isArray(value)) return value.map(_sanitizeValue)
  if (typeof value === 'object') {
    // Convert objects to strings to prevent "Objects are not valid as React child" errors
    return JSON.stringify(value)
  }
  return value
}

export default async function HuggingFaceModelPage({ params }: Props) {
  const { slug } = await params
  const modelId = slugToModelId(slug)

  try {
    // TEAM-464: Use HF model for complete data
    const hfModel = await getHuggingFaceModel(modelId)

    // TEAM-464: No README fetching - not available in current API

    // Convert to display format
    const parts = hfModel.id.split('/')
    const name = parts.length >= 2 ? parts[1] : hfModel.id
    const author = parts.length >= 2 ? parts[0] : hfModel.author

    // Calculate total size
    const totalBytes = hfModel.siblings?.reduce((sum: number, file: { size?: number; rfilename: string }) => sum + (file.size || 0), 0) || 0
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
      config: hfModel.config ? {
        architectures: hfModel.config.architectures,
        model_type: hfModel.config.model_type,
        tokenizer_config: hfModel.config.tokenizer_config ? {
          unk_token: typeof hfModel.config.tokenizer_config.unk_token === 'string' ? hfModel.config.tokenizer_config.unk_token : undefined,
          sep_token: typeof hfModel.config.tokenizer_config.sep_token === 'string' ? hfModel.config.tokenizer_config.sep_token : undefined,
          pad_token: typeof hfModel.config.tokenizer_config.pad_token === 'string' ? hfModel.config.tokenizer_config.pad_token : undefined,
          cls_token: hfModel.config.tokenizer_config.cls_token,
          mask_token: hfModel.config.tokenizer_config.mask_token,
          bos_token: hfModel.config.tokenizer_config.bos_token,
          eos_token: hfModel.config.tokenizer_config.eos_token,
          chat_template: hfModel.config.tokenizer_config.chat_template,
        } : undefined,
      } : undefined,
      cardData: hfModel.cardData,
      transformersInfo: hfModel.transformersInfo,
      inference: hfModel.inference,
      safetensors: hfModel.safetensors,
      siblings: hfModel.siblings?.map((s: { size?: number; rfilename: string }) => ({ filename: s.rfilename, size: s.size || 0 })),
      createdAt: hfModel.createdAt,
      lastModified: hfModel.lastModified,
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
