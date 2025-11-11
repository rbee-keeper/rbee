// TEAM-460: HuggingFace model detail page (migrated from /models/[slug])
// TEAM-415: SSG model detail page with slugified URLs
// TEAM-410: Added compatibility integration
// TEAM-413: Added InstallButton integration
// TEAM-475: SSR - fetches model data at request time, no manifest generation
// TEAM-475: Added Next.js caching (revalidate every 1 hour)
import { getRawHuggingFaceModel } from '@rbee/marketplace-node'
import type { Metadata } from 'next'
import { unstable_cache } from 'next/cache'
import { notFound } from 'next/navigation'
import { ModelDetailWithInstall } from '@/components/ModelDetailWithInstall'
import { slugToModelId } from '@/lib/slugify'

// TEAM-475: Cache model data for 1 hour
export const revalidate = 3600

interface Props {
  params: Promise<{ slug: string }>
}

// TEAM-475: No generateStaticParams - SSR renders on-demand

// TEAM-475: Cached metadata fetching
const getCachedHFModel = unstable_cache(
  async (modelId: string) => getRawHuggingFaceModel(modelId),
  ['hf-model'],
  { revalidate: 3600, tags: ['huggingface-models'] }
)

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params
  const modelId = slugToModelId(slug)

  try {
    const hfModel = await getCachedHFModel(modelId)
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
    // TEAM-475: Use raw HF model for complete data (cached)
    const hfModel = await getCachedHFModel(modelId)

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
          cls_token: typeof hfModel.config.tokenizer_config.cls_token === 'string' ? hfModel.config.tokenizer_config.cls_token : undefined,
          mask_token: typeof hfModel.config.tokenizer_config.mask_token === 'string' ? hfModel.config.tokenizer_config.mask_token : undefined,
          bos_token: typeof hfModel.config.tokenizer_config.bos_token === 'string' ? hfModel.config.tokenizer_config.bos_token : undefined,
          eos_token: typeof hfModel.config.tokenizer_config.eos_token === 'string' ? hfModel.config.tokenizer_config.eos_token : undefined,
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
