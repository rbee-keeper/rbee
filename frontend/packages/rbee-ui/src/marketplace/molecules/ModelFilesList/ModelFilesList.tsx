// TEAM-405: Model files list component
//! Reusable component for displaying model files/siblings

import { Card, CardContent, CardHeader, CardTitle, Badge, ScrollArea } from '@rbee/ui/atoms'
import { FileText, FileCode, FileJson, File } from 'lucide-react'

export interface ModelFile {
  rfilename: string
}

export interface ModelFilesListProps {
  files: ModelFile[]
  title?: string
  maxHeight?: string
  className?: string
}

const getFileIcon = (filename: string) => {
  if (filename.endsWith('.json')) return FileJson
  if (filename.endsWith('.py') || filename.endsWith('.js') || filename.endsWith('.ts')) return FileCode
  if (filename.endsWith('.md') || filename.endsWith('.txt')) return FileText
  return File
}

const getFileExtension = (filename: string) => {
  const parts = filename.split('.')
  return parts.length > 1 ? parts[parts.length - 1].toUpperCase() : 'FILE'
}

/**
 * Filter files to show only essential ones for rbee installation
 */
const filterEssentialFiles = (files: ModelFile[]): ModelFile[] => {
  const essentialPatterns = [
    /^model.*\.(safetensors|bin|gguf)$/i,  // Model weights
    /^config\.json$/i,                      // Model config
    /^tokenizer.*\.json$/i,                 // Tokenizer files
    /^generation_config\.json$/i,           // Generation config
    /^special_tokens_map\.json$/i,          // Special tokens
    /^vocab\.(json|txt)$/i,                 // Vocabulary
    /^merges\.txt$/i,                       // BPE merges
  ]
  
  return files.filter(file => 
    essentialPatterns.some(pattern => pattern.test(file.rfilename))
  )
}

/**
 * Card for displaying list of essential model files
 * Shows only files needed for rbee installation (weights, config, tokenizer)
 * 
 * @example
 * ```tsx
 * <ModelFilesList
 *   title="Essential Files"
 *   files={[
 *     { rfilename: 'config.json' },
 *     { rfilename: 'model.safetensors' },
 *     { rfilename: 'tokenizer.json' }
 *   ]}
 * />
 * ```
 */
export function ModelFilesList({ 
  files, 
  title = 'Essential Files', 
  maxHeight = '300px',
  className 
}: ModelFilesListProps) {
  const essentialFiles = filterEssentialFiles(files)
  
  if (essentialFiles.length === 0) return null

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>{title}</CardTitle>
          <Badge variant="secondary">{essentialFiles.length} files</Badge>
        </div>
      </CardHeader>
      <CardContent>
        <ScrollArea style={{ maxHeight }}>
          <div className="space-y-2">
            {essentialFiles.map((file, index) => {
              const Icon = getFileIcon(file.rfilename)
              const ext = getFileExtension(file.rfilename)
              
              return (
                <div
                  key={index}
                  className="flex items-center gap-3 p-2 rounded-md hover:bg-muted/50 transition-colors"
                >
                  <Icon className="size-4 text-muted-foreground flex-shrink-0" />
                  <span className="text-sm font-mono flex-1 truncate">
                    {file.rfilename}
                  </span>
                  <Badge variant="outline" className="text-xs">
                    {ext}
                  </Badge>
                </div>
              )
            })}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}
