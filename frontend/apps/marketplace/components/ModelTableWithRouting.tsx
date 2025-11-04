'use client'

import { ModelTable } from '@rbee/ui/marketplace'
import type { ModelTableItem } from '@rbee/ui/marketplace'
import { useRouter } from 'next/navigation'
import { modelIdToSlug } from '@/lib/slugify'

interface ModelTableWithRoutingProps {
  models: ModelTableItem[]
}

/**
 * Client wrapper for ModelTable that handles routing with slugified URLs
 * Converts model IDs to URL-friendly slugs for better SEO
 */
export function ModelTableWithRouting({ models }: ModelTableWithRoutingProps) {
  const router = useRouter()
  
  const handleModelClick = (modelId: string) => {
    const slug = modelIdToSlug(modelId)
    router.push(`/models/${slug}`)
  }
  
  return <ModelTable models={models} onModelClick={handleModelClick} />
}
