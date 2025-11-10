// TEAM-464: Datasets Used display component
// Shows which datasets were used to train the model

import { Badge, Card, CardContent, CardHeader, CardTitle } from '@rbee/ui/atoms'
import { Database, ExternalLink } from 'lucide-react'

export interface DatasetsUsedCardProps {
  /** List of dataset names from cardData.datasets */
  datasets?: string[]
}

/**
 * Display datasets used to train the model
 * 
 * Matches the "Datasets used to train" section on HuggingFace model pages
 */
export function DatasetsUsedCard({ datasets }: DatasetsUsedCardProps) {
  if (!datasets || datasets.length === 0) {
    return null
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Database className="size-5" />
          <CardTitle>Datasets used to train</CardTitle>
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex flex-wrap gap-2">
          {datasets.map((dataset) => (
            <a
              key={dataset}
              href={`https://huggingface.co/datasets/${dataset}`}
              target="_blank"
              rel="noopener noreferrer"
              className="group"
            >
              <Badge
                variant="outline"
                className="text-sm hover:bg-muted transition-colors flex items-center gap-1"
              >
                {dataset}
                <ExternalLink className="size-3 opacity-0 group-hover:opacity-100 transition-opacity" />
              </Badge>
            </a>
          ))}
        </div>
        {datasets.length > 10 && (
          <p className="text-xs text-muted-foreground mt-3 pt-3 border-t">
            Showing {datasets.length} datasets
          </p>
        )}
      </CardContent>
    </Card>
  )
}
