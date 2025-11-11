// TEAM-460: HuggingFace models marketplace page (migrated from /models)
// TEAM-475: SSR initial load, then client-side filtering (SPA experience)
// TEAM-475: Fetches ALL compatible models once, client filters them
// TEAM-475: Added compatibility filtering - only shows models that work with LLM_worker

import { getCompatibleHuggingFaceModels } from '@rbee/marketplace-node'
import type { Metadata } from 'next'
import { HFFilterPage } from './HFFilterPage'

export const metadata: Metadata = {
  title: 'HuggingFace LLM Models | rbee Marketplace',
  description: 'Browse compatible language models from HuggingFace. Real-time data, updated every 5 minutes.',
}

// TEAM-XXX: Force dynamic rendering to ensure client-side filtering works correctly
// Without this, Next.js may treat this route as static and cause SSR on filter changes
export const dynamic = 'force-dynamic'

// TEAM-475: SSR - fetch ALL compatible models once on initial load
// Client-side filtering handles filter changes (no re-fetch)
// Only fetches models that work with LLM_worker (llama, mistral, phi, qwen, gemma)
export default async function HuggingFaceModelsPage() {
  const FETCH_LIMIT = 300

  console.log(`[SSR] Fetching compatible HuggingFace models (filtered by LLM_worker compatibility)`)

  const hfModels = await getCompatibleHuggingFaceModels({ limit: FETCH_LIMIT })

  console.log(`[SSR] Rendering ${hfModels.length} HuggingFace models`)

  // Normalize models for client component
  const models = hfModels.map((model) => {
    const m = model as unknown as Record<string, unknown>
    return {
      id: m.id as string,
      name: (m.name as string) || (m.id as string),
      description: (m.description as string) || `${(m.author as string) || 'Community'} model`,
      author: m.author as string | undefined,
      downloads: (m.downloads as number) ?? 0,
      likes: (m.likes as number) ?? 0,
      tags: (m.tags as string[] | undefined)?.slice(0, 10) ?? [],
    }
  })

  // TEAM-475: Pass SSR data to client component
  // Client component handles all filtering without re-fetching
  return <HFFilterPage initialModels={models} />
}
