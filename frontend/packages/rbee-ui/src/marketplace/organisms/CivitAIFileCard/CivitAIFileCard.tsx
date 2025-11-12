// TEAM-463: Premium file download card for CivitAI models
// TEAM-478: Enhanced with better visual design and file type detection
// Beautiful file cards with download buttons and size display

'use client'

import { Badge, Button, Card } from '@rbee/ui/atoms'
import { cn } from '@rbee/ui/utils'
import { CheckCircle2, Download, FileArchive, FileCode, FileText, HardDrive } from 'lucide-react'

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

function getFileExtension(filename: string): string {
  const parts = filename.split('.')
  return parts.length > 1 ? `.${parts[parts.length - 1]}` : '.safetensors'
}

function getFileIcon(filename: string) {
  const ext = getFileExtension(filename).toLowerCase()
  if (ext === '.safetensors' || ext === '.ckpt' || ext === '.pt' || ext === '.pth') return HardDrive
  if (ext === '.yaml' || ext === '.yml' || ext === '.json') return FileCode
  if (ext === '.zip' || ext === '.tar' || ext === '.gz') return FileArchive
  return FileText
}

export function CivitAIFileCard({ files, className }: CivitAIFileCardProps) {
  if (files.length === 0) {
    return null
  }

  return (
    <Card className={cn('p-6 space-y-5 shadow-lg', className)}>
      {/* Header */}
      <div className="flex items-center justify-between pb-2 border-b">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-primary/10">
            <Download className="size-5 text-primary" />
          </div>
          <div>
            <h3 className="font-semibold text-lg">
              {files.length} {files.length === 1 ? 'File' : 'Files'}
            </h3>
            <p className="text-xs text-muted-foreground">
              {formatFileSize(files.reduce((sum, f) => sum + f.sizeKb, 0))} total size
            </p>
          </div>
        </div>
        <Badge variant="secondary" className="text-xs font-mono">
          {files.filter((f) => f.primary).length > 0 ? '1 primary' : 'All files'}
        </Badge>
      </div>

      {/* Files List - Simple Table Layout */}
      <div className="space-y-3">
        {files.map((file) => {
          const FileIcon = getFileIcon(file.name)
          const fileExt = getFileExtension(file.name)
          
          return (
            <div
              key={file.id}
              className="group rounded-lg border border-border bg-card hover:bg-accent/5 transition-colors"
            >
              <div className="p-4 space-y-3">
                {/* Row 1: Icon + Filename + Primary Badge */}
                <div className="flex items-center gap-3">
                  <FileIcon className="size-5 text-muted-foreground shrink-0" />
                  <p className="font-medium text-sm truncate flex-1 min-w-0" title={file.name}>
                    {file.name}
                  </p>
                  {file.primary && (
                    <Badge variant="default" className="flex items-center gap-1 text-xs shrink-0">
                      <CheckCircle2 className="size-3" />
                      Primary
                    </Badge>
                  )}
                </div>

                {/* Row 2: Badges + Download Button */}
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="font-mono text-xs">
                      {formatFileSize(file.sizeKb)}
                    </Badge>
                    <Badge variant="secondary" className="font-mono text-xs">
                      {fileExt}
                    </Badge>
                  </div>
                  <Button size="sm" className="shrink-0" asChild>
                    <a href={file.downloadUrl} target="_blank" rel="noopener noreferrer" title="Download file">
                      <Download className="size-4" />
                    </a>
                  </Button>
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </Card>
  )
}
