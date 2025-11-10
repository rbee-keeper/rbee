// TEAM-405: Model files list component
// TEAM-463: Type contract - must match artifacts-contract ModelFile type
//! Reusable component for displaying model files/siblings
//! 
//! ⚠️ TYPE CONTRACT: This ModelFile interface MUST match the canonical Rust type:
//! `bin/97_contracts/artifacts-contract/src/model/mod.rs::ModelFile`
//! 
//! The type flows as:
//! artifacts-contract (source of truth)
//!   ↓ re-exported by
//! marketplace-sdk
//!   ↓ generates TypeScript types for
//! keeper GUI (auto-generated bindings.ts)
//!   
//! This manual definition is for Next.js marketplace which can't import Tauri types.
//! If the contract type changes, this interface must be updated manually.

import { Badge, Card, CardContent, CardHeader, CardTitle, ScrollArea } from '@rbee/ui/atoms'
import { File, FileCode, FileJson, FileText } from 'lucide-react'

/**
 * Model file information
 * 
 * ⚠️ MUST MATCH: artifacts-contract/src/model/mod.rs::ModelFile
 */
export interface ModelFile {
  /** File name (relative path in repo) */
  filename: string
  /** File size in bytes (optional) */
  size?: number | null
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
    /^model.*\.(safetensors|bin|gguf)$/i, // Model weights
    /^config\.json$/i, // Model config
    /^tokenizer.*\.json$/i, // Tokenizer files
    /^generation_config\.json$/i, // Generation config
    /^special_tokens_map\.json$/i, // Special tokens
    /^vocab\.(json|txt)$/i, // Vocabulary
    /^merges\.txt$/i, // BPE merges
  ]

  return files.filter((file) => essentialPatterns.some((pattern) => pattern.test(file.filename)))
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
 *     { filename: 'config.json' },
 *     { filename: 'model.safetensors' },
 *     { filename: 'tokenizer.json' }
 *   ]}
 * />
 * ```
 */
export function ModelFilesList({
  files,
  title = 'Essential Files',
  maxHeight = '300px',
  className,
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
            {essentialFiles.map((file) => {
              const Icon = getFileIcon(file.filename)
              const ext = getFileExtension(file.filename)

              return (
                <div key={file.filename} className="flex items-center gap-3 p-2 rounded-md hover:bg-muted/50 transition-colors">
                  <Icon className="size-4 text-muted-foreground flex-shrink-0" />
                  <span className="text-sm font-mono flex-1 truncate">{file.filename}</span>
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
