// TEAM-478: CivitAI model detail page - Enhanced with full API data
// Fetches raw CivitAI API data to show versions, all images, files, stats
// Uses CivitAIModelDetail template (3-column design with version selector)

import type { CivitAIModel } from '@rbee/marketplace-core'
import { CivitAIModelDetail } from '@rbee/ui/marketplace'
import type { Metadata } from 'next'
import { notFound } from 'next/navigation'

// Cache for 1 hour
export const revalidate = 3600

interface Props {
  params: Promise<{ slug: string }>
}

// Helper to convert slug to model ID
function slugToModelId(slug: string): number {
  // Slug format: "civitai-12345-model-name" -> 12345
  // OR just "12345" -> 12345
  const decoded = decodeURIComponent(slug)

  // Try to extract number from slug
  const match = decoded.match(/(\d+)/)
  if (match?.[1]) {
    return parseInt(match[1], 10)
  }

  // Fallback: try to parse as number
  const id = parseInt(decoded, 10)
  if (!Number.isNaN(id)) {
    return id
  }

  throw new Error(`Invalid CivitAI model ID: ${slug}`)
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params

  try {
    const modelId = slugToModelId(slug)
    const rawModel = await fetchCivitAIModelRaw(modelId)

    const description = rawModel.description
      ? rawModel.description
          .replace(/<[^>]*>/g, ' ')
          .replace(/\s+/g, ' ')
          .trim()
          .slice(0, 200)
      : `${rawModel.name} - ${rawModel.stats.downloadCount.toLocaleString()} downloads`

    const primaryImage = rawModel.modelVersions?.[0]?.images?.[0]?.url

    return {
      title: `${rawModel.name} | CivitAI Model`,
      description,
      openGraph: {
        title: `${rawModel.name} | CivitAI Model`,
        description,
        images: primaryImage ? [{ url: primaryImage }] : undefined,
      },
    }
  } catch {
    return { title: 'Model Not Found' }
  }
}

export default async function CivitAIModelDetailPage({ params }: Props) {
  const { slug } = await params

  try {
    const modelId = slugToModelId(slug)

    // TEAM-478: Fetch raw CivitAI API data (includes all versions, images, files)
    const rawModel = await fetchCivitAIModelRaw(modelId)

    // TEAM-478: Convert to CivitAIModelDetailProps with full data
    const primaryVersion = rawModel.modelVersions?.[0]
    const primaryFile = primaryVersion?.files?.find((f) => f.primary) ?? primaryVersion?.files?.[0]

    const civitaiModelData = {
      id: String(rawModel.id),
      name: rawModel.name,
      description: rawModel.description || '',
      author: rawModel.creator?.username || 'Unknown',
      downloads: rawModel.stats?.downloadCount ?? 0,
      likes: rawModel.stats?.favoriteCount ?? 0,
      rating: rawModel.stats?.rating ?? 0,
      thumbsUpCount: rawModel.stats?.thumbsUpCount,
      commentCount: rawModel.stats?.commentCount,
      ratingCount: rawModel.stats?.ratingCount,
      size: primaryFile?.sizeKB ? formatBytes(primaryFile.sizeKB * 1024) : 'Unknown',
      tags: rawModel.tags ?? [],
      type: rawModel.type,
      baseModel: primaryVersion?.baseModel || 'Unknown',
      version: primaryVersion?.name || 'Latest',
      images:
        primaryVersion?.images?.map((img) => ({
          url: img.url,
          nsfw: img.nsfwLevel > 1,
          width: img.width,
          height: img.height,
        })) ?? [],
      files:
        primaryVersion?.files?.map((file) => ({
          name: file.name,
          id: file.id,
          sizeKb: file.sizeKB,
          downloadUrl: file.downloadUrl || '',
          primary: file.primary ?? false,
        })) ?? [],
      ...(primaryVersion?.trainedWords ? { trainedWords: primaryVersion.trainedWords } : {}),
      allowCommercialUse: rawModel.allowCommercialUse?.join(', ') || 'Unknown',
      externalUrl: `https://civitai.com/models/${rawModel.id}`,
      externalLabel: 'View on CivitAI',
      // TEAM-478: Pass all versions for version selector
      versions:
        rawModel.modelVersions?.map((version) => ({
          id: version.id,
          name: version.name,
          ...(version.baseModel ? { baseModel: version.baseModel } : {}),
          ...(version.description ? { description: version.description } : {}),
          createdAt: version.createdAt,
          updatedAt: version.updatedAt,
          ...(version.publishedAt ? { publishedAt: version.publishedAt } : {}),
          ...(version.trainedWords ? { trainedWords: version.trainedWords } : {}),
          images:
            version.images?.map((img) => ({
              url: img.url,
              nsfw: img.nsfwLevel > 1,
              width: img.width,
              height: img.height,
            })) ?? [],
          files:
            version.files?.map((file) => ({
              name: file.name,
              id: file.id,
              sizeKb: file.sizeKB,
              downloadUrl: file.downloadUrl || '',
              primary: file.primary ?? false,
            })) ?? [],
          ...(version.downloadUrl ? { downloadUrl: version.downloadUrl } : {}),
          ...(version.stats ? { stats: version.stats } : {}),
        })) ?? [],
      ...(primaryVersion?.publishedAt ? { publishedAt: primaryVersion.publishedAt } : {}),
      ...(primaryVersion?.updatedAt ? { updatedAt: primaryVersion.updatedAt } : {}),
    }

    return (
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <CivitAIModelDetail model={civitaiModelData} />
      </div>
    )
  } catch (error) {
    console.error('[CivitAI Detail] Error:', error)
    notFound()
  }
}

// TEAM-478: Fetch raw CivitAI API data (not normalized)
async function fetchCivitAIModelRaw(modelId: number): Promise<CivitAIModel> {
  const url = `https://civitai.com/api/v1/models/${modelId}`
  const response = await fetch(url, { next: { revalidate: 3600 } })

  if (!response.ok) {
    throw new Error(`CivitAI API error: ${response.status}`)
  }

  return response.json() as Promise<CivitAIModel>
}

// Helper function
function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${Math.round((bytes / k ** i) * 100) / 100} ${sizes[i]}`
}
