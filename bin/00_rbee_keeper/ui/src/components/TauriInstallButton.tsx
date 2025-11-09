// TEAM-413: Install button for Tauri (Keeper GUI)
// Actually downloads models/workers (unlike web version which triggers rbee:// protocol)

import { Button } from '@rbee/ui/atoms'
import { invoke } from '@tauri-apps/api/core'
import { CheckCircle2Icon, DownloadIcon, Loader2Icon } from 'lucide-react'
import { useState } from 'react'
import { useDownloadStore } from '@/store/downloadStore'

interface TauriInstallButtonProps {
  modelId?: string
  workerId?: string
  hiveId?: string
  variant?: 'default' | 'outline' | 'ghost'
  size?: 'default' | 'sm' | 'lg' | 'icon' | 'icon-sm' | 'icon-lg'
  className?: string
}

export function TauriInstallButton({
  modelId,
  workerId,
  hiveId = 'localhost',
  variant = 'default',
  size = 'lg',
  className,
}: TauriInstallButtonProps) {
  const [isSubmitting, setIsSubmitting] = useState(false)
  const { startDownload, downloads } = useDownloadStore()

  // Check if already downloading
  const itemId = modelId || workerId || ''
  const existingDownload = downloads.find((d) => d.name === itemId)
  const isDownloading = existingDownload?.status === 'downloading'
  const isComplete = existingDownload?.status === 'complete'

  const handleInstall = async () => {
    if (!modelId && !workerId) {
      console.error('TauriInstallButton: No modelId or workerId provided')
      return
    }

    setIsSubmitting(true)

    try {
      if (modelId) {
        // Download model
        const jobId = await invoke<string>('model_download', {
          hiveId,
          modelId,
        })

        // Add to download tracker
        startDownload(jobId, modelId, 'model')
        console.log(`[TauriInstallButton] Model download started: ${jobId}`)
      } else if (workerId) {
        // Download worker
        const jobId = await invoke<string>('worker_download', {
          hiveId,
          workerId,
        })

        // Add to download tracker
        startDownload(jobId, workerId, 'worker')
        console.log(`[TauriInstallButton] Worker download started: ${jobId}`)
      }
    } catch (error) {
      console.error('[TauriInstallButton] Download failed:', error)
      // TODO: Show error toast
    } finally {
      setIsSubmitting(false)
    }
  }

  // Show different states
  if (isComplete) {
    return (
      <Button variant={variant} size={size} className={className} disabled>
        <CheckCircle2Icon className="w-4 h-4 mr-2" />
        Installed
      </Button>
    )
  }

  if (isDownloading) {
    return (
      <Button variant={variant} size={size} className={className} disabled>
        <Loader2Icon className="w-4 h-4 mr-2 animate-spin" />
        Downloading {existingDownload.percentage ? `${existingDownload.percentage.toFixed(0)}%` : '...'}
      </Button>
    )
  }

  return (
    <Button variant={variant} size={size} className={className} onClick={handleInstall} disabled={isSubmitting}>
      {isSubmitting ? (
        <>
          <Loader2Icon className="w-4 h-4 mr-2 animate-spin" />
          Starting...
        </>
      ) : (
        <>
          <DownloadIcon className="w-4 h-4 mr-2" />
          Download {modelId ? 'Model' : 'Worker'}
        </>
      )}
    </Button>
  )
}
