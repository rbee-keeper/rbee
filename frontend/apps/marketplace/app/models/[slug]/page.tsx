// TEAM-427: Legacy redirect route for /models/[slug]
// Auto-detects provider (HuggingFace vs CivitAI) and redirects to correct path

import { redirect } from 'next/navigation'
import { listHuggingFaceModels, getCompatibleCivitaiModels } from '@rbee/marketplace-node'
import { modelIdToSlug } from '@/lib/slugify'

interface PageProps {
  params: Promise<{ slug: string }>
}

export async function generateStaticParams() {
  // Generate params for all HuggingFace models
  const hfModels = await listHuggingFaceModels({ limit: 100 })
  const hfParams = hfModels.map((model) => ({
    slug: modelIdToSlug((model as { id: string }).id),
  }))

  // Generate params for all CivitAI models
  const civitaiModels = await getCompatibleCivitaiModels()
  const civitaiParams = civitaiModels.map((model) => ({
    slug: modelIdToSlug(model.id),
  }))

  return [...hfParams, ...civitaiParams]
}

export default async function LegacyModelRedirect({ params }: PageProps) {
  const { slug } = await params
  
  // CivitAI models have "civitai-" prefix in their slugs
  // Example: civitai-4201, civitai-133005
  if (slug.startsWith('civitai-')) {
    redirect(`/models/civitai/${slug}`)
  }
  
  // Everything else is HuggingFace
  // Example: sentence-transformers--all-minilm-l6-v2
  redirect(`/models/huggingface/${slug}`)
}
