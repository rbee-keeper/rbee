// TEAM-405: Client wrapper for interactive table
'use client'

import { ModelListTableTemplate } from '@rbee/ui/marketplace'
import type { ModelTableItem } from '@rbee/ui/marketplace'
import { useRouter } from 'next/navigation'

interface ModelListClientProps {
  initialModels: ModelTableItem[]
}

export function ModelListClient({ initialModels }: ModelListClientProps) {
  const router = useRouter()
  
  return (
    <ModelListTableTemplate
      models={initialModels}
      onModelClick={(id) => router.push(`/models/${encodeURIComponent(id)}`)}
    />
  )
}
