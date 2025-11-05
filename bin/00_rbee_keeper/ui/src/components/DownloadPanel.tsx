// TEAM-413: Download tracking panel for models and workers
// Shows active downloads with progress bars (like ML Studio / Ollama)

import { DownloadIcon, CheckCircle2Icon, XCircleIcon, Loader2Icon } from 'lucide-react'
import { Progress } from '@rbee/ui/atoms'

export interface Download {
  id: string
  name: string
  type: 'model' | 'worker'
  status: 'downloading' | 'complete' | 'failed' | 'cancelled'
  bytesDownloaded: number
  totalSize: number | null
  percentage: number | null
  speed: string | null // e.g., "2.5 MB/s"
  eta: string | null // e.g., "2m 30s"
  error?: string
}

interface DownloadPanelProps {
  downloads: Download[]
  onCancel?: (id: string) => void
  onRetry?: (id: string) => void
  onClear?: (id: string) => void
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`
}

function DownloadItem({ download, onCancel, onRetry, onClear }: {
  download: Download
  onCancel?: (id: string) => void
  onRetry?: (id: string) => void
  onClear?: (id: string) => void
}) {
  const isActive = download.status === 'downloading'
  const isComplete = download.status === 'complete'
  const isFailed = download.status === 'failed' || download.status === 'cancelled'

  return (
    <div className="p-3 rounded-lg border border-border bg-card space-y-2">
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            {isActive && <Loader2Icon className="w-3 h-3 animate-spin text-primary flex-shrink-0" />}
            {isComplete && <CheckCircle2Icon className="w-3 h-3 text-green-500 flex-shrink-0" />}
            {isFailed && <XCircleIcon className="w-3 h-3 text-destructive flex-shrink-0" />}
            <span className="text-sm font-medium truncate">{download.name}</span>
          </div>
          <span className="text-xs text-muted-foreground capitalize">{download.type}</span>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1">
          {isActive && onCancel && (
            <button
              onClick={() => onCancel(download.id)}
              className="p-1 rounded hover:bg-muted transition-colors"
              title="Cancel download"
            >
              <XCircleIcon className="w-4 h-4" />
            </button>
          )}
          {isFailed && onRetry && (
            <button
              onClick={() => onRetry(download.id)}
              className="p-1 rounded hover:bg-muted transition-colors text-xs"
              title="Retry download"
            >
              Retry
            </button>
          )}
          {(isComplete || isFailed) && onClear && (
            <button
              onClick={() => onClear(download.id)}
              className="p-1 rounded hover:bg-muted transition-colors text-xs"
              title="Clear from list"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Progress Bar */}
      {isActive && (
        <div className="space-y-1">
          <Progress 
            value={download.percentage || 0} 
            className="h-1.5"
          />
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>
              {download.percentage !== null 
                ? `${download.percentage.toFixed(1)}%` 
                : 'Calculating...'}
            </span>
            {download.totalSize && (
              <span>
                {formatBytes(download.bytesDownloaded)} / {formatBytes(download.totalSize)}
              </span>
            )}
          </div>
          {(download.speed || download.eta) && (
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              {download.speed && <span>{download.speed}</span>}
              {download.eta && <span>ETA: {download.eta}</span>}
            </div>
          )}
        </div>
      )}

      {/* Complete State */}
      {isComplete && download.totalSize && (
        <div className="text-xs text-muted-foreground">
          {formatBytes(download.totalSize)} downloaded
        </div>
      )}

      {/* Error State */}
      {isFailed && download.error && (
        <div className="text-xs text-destructive">
          {download.error}
        </div>
      )}
    </div>
  )
}

export function DownloadPanel({ downloads, onCancel, onRetry, onClear }: DownloadPanelProps) {
  const activeDownloads = downloads.filter(d => d.status === 'downloading')
  const completedDownloads = downloads.filter(d => d.status === 'complete')
  const failedDownloads = downloads.filter(d => d.status === 'failed' || d.status === 'cancelled')

  if (downloads.length === 0) {
    return null // Don't show panel if no downloads
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 px-2">
        <DownloadIcon className="w-4 h-4 text-muted-foreground" />
        <h3 className="text-xs font-semibold font-serif text-muted-foreground uppercase tracking-wider">
          Downloads
        </h3>
        {activeDownloads.length > 0 && (
          <span className="text-xs text-muted-foreground">
            ({activeDownloads.length} active)
          </span>
        )}
      </div>

      <div className="space-y-2">
        {/* Active Downloads */}
        {activeDownloads.map(download => (
          <DownloadItem
            key={download.id}
            download={download}
            onCancel={onCancel}
          />
        ))}

        {/* Failed Downloads */}
        {failedDownloads.map(download => (
          <DownloadItem
            key={download.id}
            download={download}
            onRetry={onRetry}
            onClear={onClear}
          />
        ))}

        {/* Completed Downloads (show last 3) */}
        {completedDownloads.slice(0, 3).map(download => (
          <DownloadItem
            key={download.id}
            download={download}
            onClear={onClear}
          />
        ))}
      </div>
    </div>
  )
}
