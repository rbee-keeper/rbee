// TEAM-482: Worker List Card - reusable component for worker listings
// Simple card design with image support
// TEAM-501: Fixed 1:1 aspect ratio for images, fixed card sizing

'use client'

import { Badge } from '@rbee/ui/atoms/Badge'
import { cn } from '@rbee/ui/utils'
import { Cpu } from 'lucide-react'
import Image from 'next/image'
import Link from 'next/link'

export type WorkerType = 'cpu' | 'cuda' | 'metal' | 'rocm'

export interface WorkerListCardProps {
  worker: {
    id: string
    name: string
    description: string
    version: string
    backends: WorkerType[] // TEAM-501: Show all supported backends
    imageUrl?: string
  }
  href?: string
  className?: string
}

const workerTypeConfig = {
  cpu: { label: 'CPU', variant: 'secondary' as const },
  cuda: { label: 'CUDA', variant: 'default' as const },
  metal: { label: 'Metal', variant: 'accent' as const },
  rocm: { label: 'ROCm', variant: 'destructive' as const },
}

export function WorkerListCard({ worker, href, className }: WorkerListCardProps) {
  // TEAM-501: No longer need single type config, we show all backends

  const content = (
    <div
      className={cn(
        'border border-border rounded-lg overflow-hidden hover:border-primary/50 transition-colors bg-card h-full flex flex-col',
        href && 'cursor-pointer',
        className,
      )}
    >
      {/* Worker image or placeholder - 1:1 aspect ratio */}
      <div className="relative w-full aspect-square bg-muted flex items-center justify-center">
        {worker.imageUrl ? (
          <Image
            src={worker.imageUrl}
            alt={worker.name}
            fill
            className="object-cover"
            sizes="(max-width: 768px) 100vw, (max-width: 1200px) 33vw, 25vw"
          />
        ) : (
          <Cpu className="size-16 text-muted-foreground/30" />
        )}
      </div>

      {/* Worker info */}
      <div className="p-4 flex-grow flex flex-col">
        {/* Name and version */}
        <div className="mb-2">
          <h3 className="font-bold text-lg truncate">{worker.name}</h3>
          <p className="text-xs text-muted-foreground">v{worker.version}</p>
        </div>

        {/* Backend badges - subtle outline style */}
        <div className="flex flex-wrap gap-1 mb-2">
          {worker.backends.map((backend) => {
            const config = workerTypeConfig[backend] || workerTypeConfig.cpu
            return (
              <Badge key={backend} variant="outline" className="text-xs">
                {config.label}
              </Badge>
            )
          })}
        </div>

        {/* Description */}
        <p className="text-sm text-muted-foreground line-clamp-2 mt-auto">{worker.description}</p>
      </div>
    </div>
  )

  if (href) {
    return (
      <Link href={href} className="block">
        {content}
      </Link>
    )
  }

  return content
}
