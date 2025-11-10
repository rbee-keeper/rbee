// TEAM-463: Premium file download card for CivitAI models
// Beautiful file cards with download buttons and size display

'use client'

import { Badge, Button, Card } from '@rbee/ui/atoms'
import { Download, FileText, CheckCircle2 } from 'lucide-react'
import { cn } from '@rbee/ui/utils'

export interface CivitAIFile {
  name: string
  id: number
  sizeKb: number
  downloadUrl: string
  primary: boolean
}

export interface CivitAIFileCardProps {
  files: CivitAIFile[]
  className?: string
}

function formatFileSize(sizeKb: number): string {
  if (sizeKb < 1024) return `${sizeKb.toFixed(0)} KB`
  const sizeMb = sizeKb / 1024
  if (sizeMb < 1024) return `${sizeMb.toFixed(2)} MB`
  const sizeGb = sizeMb / 1024
  return `${sizeGb.toFixed(2)} GB`
}

export function CivitAIFileCard({ files, className }: CivitAIFileCardProps) {
  if (files.length === 0) {
    return null
  }

  return (
    <Card className={cn('p-6 space-y-4', className)}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FileText className="size-5 text-primary" />
          <h3 className="font-semibold text-lg">
            {files.length} {files.length === 1 ? 'File' : 'Files'}
          </h3>
        </div>
        <Badge variant="outline" className="text-xs">
          {formatFileSize(files.reduce((sum, f) => sum + f.sizeKb, 0))} total
        </Badge>
      </div>

      <div className="space-y-3">
        {files.map((file) => (
          <div
            key={file.id}
            className="group relative overflow-hidden rounded-lg border bg-card p-4 transition-all duration-200 hover:shadow-md hover:border-primary/50"
          >
            {/* Background Gradient on Hover */}
            <div className="absolute inset-0 bg-gradient-to-r from-primary/5 to-transparent opacity-0 transition-opacity group-hover:opacity-100" />

            <div className="relative flex items-start justify-between gap-3">
              <div className="flex-1 min-w-0 space-y-2">
                <div className="flex items-center gap-2">
                  <p className="font-medium text-sm truncate">{file.name}</p>
                  {file.primary && (
                    <Badge variant="default" className="text-xs flex items-center gap-1">
                      <CheckCircle2 className="size-3" />
                      Primary
                    </Badge>
                  )}
                </div>
                <div className="flex items-center gap-3 text-xs text-muted-foreground">
                  <span className="font-mono">{formatFileSize(file.sizeKb)}</span>
                  <span className="text-muted-foreground/50">•</span>
                  <span className="truncate">.safetensors</span>
                </div>
              </div>

              <Button
                size="sm"
                variant="outline"
                className="shrink-0 group-hover:bg-primary group-hover:text-primary-foreground transition-colors"
                asChild
              >
                <a href={file.downloadUrl} target="_blank" rel="noopener noreferrer">
                  <Download className="size-4 mr-2" />
                  Download
                </a>
              </Button>
            </div>

            {/* Progress Bar Effect on Hover */}
            <div className="absolute bottom-0 left-0 h-0.5 w-0 bg-primary transition-all duration-300 group-hover:w-full" />
          </div>
        ))}
      </div>
    </Card>
  )
}
