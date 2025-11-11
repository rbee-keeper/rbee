// TEAM-427: Legacy redirect route for /models/[slug]
// Auto-detects provider (HuggingFace vs CivitAI) and redirects to correct path
// TEAM-475: SSR - no manifest generation, renders on-demand

import { redirect } from 'next/navigation'

interface PageProps {
  params: Promise<{ slug: string }>
}

// TEAM-475: No generateStaticParams - SSR renders on-demand

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
