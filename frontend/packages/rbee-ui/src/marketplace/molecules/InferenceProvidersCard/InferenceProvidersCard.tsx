// TEAM-464: Inference Providers display component
// Shows which inference providers support this model

import { Badge, Card, CardContent, CardHeader, CardTitle } from '@rbee/ui/atoms'
import { ExternalLink, Zap } from 'lucide-react'

export interface InferenceProvider {
  name: string
  url: string
  status?: 'available' | 'unavailable'
}

export interface InferenceProvidersCardProps {
  /** Inference status from HuggingFace API */
  inferenceStatus?: 'warm' | 'cold' | string
  /** Library name (e.g., 'sentence-transformers') */
  libraryName?: string
  /** Transformers info */
  transformersInfo?: {
    auto_model?: string
    pipeline_tag?: string
    processor?: string
  }
}

/**
 * Display inference providers and model inference information
 * 
 * Matches the "Inference Providers" section on HuggingFace model pages
 */
export function InferenceProvidersCard({
  inferenceStatus,
  libraryName,
  transformersInfo,
}: InferenceProvidersCardProps) {
  // Don't render if no inference info available
  if (!inferenceStatus && !libraryName && !transformersInfo) {
    return null
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Zap className="size-5" />
          <CardTitle>Inference Providers</CardTitle>
          {inferenceStatus && (
            <Badge variant={inferenceStatus === 'warm' ? 'default' : 'secondary'}>
              {inferenceStatus}
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* HuggingFace Inference API */}
        <div className="flex items-start justify-between p-3 bg-muted/50 rounded-lg">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="font-medium">HF Inference API</span>
              <Badge variant="outline" className="text-xs">
                Serverless
              </Badge>
            </div>
            {libraryName && (
              <p className="text-sm text-muted-foreground">
                Using <code className="px-1 py-0.5 bg-background rounded text-xs">{libraryName}</code>
              </p>
            )}
          </div>
          <a
            href="https://huggingface.co/inference-api"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-primary hover:underline flex items-center gap-1"
          >
            View API
            <ExternalLink className="size-3" />
          </a>
        </div>

        {/* Transformers Info */}
        {transformersInfo && (
          <div className="space-y-2 pt-2 border-t">
            <h4 className="text-sm font-medium">Transformers Configuration</h4>
            <div className="grid gap-2 text-sm">
              {transformersInfo.auto_model && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Auto Model:</span>
                  <code className="px-1 py-0.5 bg-muted rounded text-xs">{transformersInfo.auto_model}</code>
                </div>
              )}
              {transformersInfo.processor && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Processor:</span>
                  <code className="px-1 py-0.5 bg-muted rounded text-xs">{transformersInfo.processor}</code>
                </div>
              )}
              {transformersInfo.pipeline_tag && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Pipeline:</span>
                  <code className="px-1 py-0.5 bg-muted rounded text-xs">{transformersInfo.pipeline_tag}</code>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Info text */}
        <p className="text-xs text-muted-foreground pt-2 border-t">
          Inference Providers let you run inference on thousands of models using a simple, unified API.
        </p>
      </CardContent>
    </Card>
  )
}
