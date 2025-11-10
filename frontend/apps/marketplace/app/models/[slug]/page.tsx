// TEAM-427: Legacy redirect route for /models/[slug]
// Auto-detects provider (HuggingFace vs CivitAI) and redirects to correct path
// TEAM-464: Using manifest-based SSG (Phase 2)

import { redirect } from 'next/navigation'
import { loadAllModels } from '@/lib/manifests'

interface PageProps {
  params: Promise<{ slug: string }>
}

export async function generateStaticParams() {
  console.log('[SSG] Generating model redirect pages from manifest')
  
  // TEAM-464: Read from manifest instead of API
  const models = await loadAllModels()
  
  console.log(`[SSG] Pre-building ${models.length} redirect pages`)

  return models.map((model) => ({
    slug: model.slug,
  }))
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
