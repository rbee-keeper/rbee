// TEAM-464: Widget Data display component
// Shows inference examples from HuggingFace widget data

import { Card, CardContent, CardHeader, CardTitle } from '@rbee/ui/atoms'
import { MessageSquare } from 'lucide-react'

export interface WidgetData {
  source_sentence?: string
  sentences?: string[]
  text?: string
}

export interface WidgetDataCardProps {
  /** Widget data from HuggingFace API */
  widgetData?: WidgetData[]
  /** Pipeline tag to customize the display */
  pipelineTag?: string
}

/**
 * Display inference widget examples
 * 
 * Shows example inputs/outputs for the model based on its pipeline type
 */
export function WidgetDataCard({ widgetData, pipelineTag }: WidgetDataCardProps) {
  if (!widgetData || widgetData.length === 0) {
    return null
  }

  const isSentenceSimilarity = pipelineTag === 'sentence-similarity'

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <MessageSquare className="size-5" />
          <CardTitle>Usage (Sentence-Transformers)</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {widgetData.map((widget, index) => (
          <div key={index} className="space-y-3">
            {/* Sentence Similarity Example */}
            {isSentenceSimilarity && widget.source_sentence && widget.sentences && (
              <div className="space-y-2">
                <div className="text-sm font-medium text-muted-foreground">Source Sentence:</div>
                <div className="p-3 bg-muted/50 rounded-lg">
                  <code className="text-sm">{widget.source_sentence}</code>
                </div>
                <div className="text-sm font-medium text-muted-foreground mt-3">Compare with:</div>
                <div className="space-y-2">
                  {widget.sentences.map((sentence, idx) => (
                    <div key={idx} className="p-3 bg-muted/30 rounded-lg border border-border/50">
                      <code className="text-sm">{sentence}</code>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Generic Text Example */}
            {widget.text && (
              <div className="p-3 bg-muted/50 rounded-lg">
                <code className="text-sm whitespace-pre-wrap">{widget.text}</code>
              </div>
            )}
          </div>
        ))}

        <p className="text-xs text-muted-foreground pt-3 border-t">
          Using this model becomes easy when you have{' '}
          <a
            href="https://www.sbert.net/docs/installation.html"
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary hover:underline"
          >
            sentence-transformers
          </a>{' '}
          installed.
        </p>
      </CardContent>
    </Card>
  )
}
