// TEAM-482: Worker List Card - reusable component for worker listings
// Simple card design with image support

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
    workerType: WorkerType
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
  const typeConfig = workerTypeConfig[worker.workerType] || workerTypeConfig.cpu

  const content = (
    <div
      className={cn(
        'border border-border rounded-lg overflow-hidden hover:border-primary/50 transition-colors bg-card h-full flex flex-col',
        href && 'cursor-pointer',
        className,
      )}
    >
      {/* Worker image or placeholder */}
      <div className="relative w-full h-40 bg-muted flex items-center justify-center">
        {worker.imageUrl ? (
          <Image
            src={worker.imageUrl}
            alt={worker.name}
            fill
            className="object-cover"
            sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
          />
        ) : (
          <Cpu className="size-16 text-muted-foreground/30" />
        )}
      </div>

      {/* Worker info */}
      <div className="p-4 flex-grow flex flex-col">
        {/* Name and type */}
        <div className="mb-2">
          <div className="flex items-center gap-2 mb-1">
            <h3 className="font-bold text-lg truncate flex-1">{worker.name}</h3>
            <Badge variant={typeConfig.variant} className="text-xs shrink-0">
              {typeConfig.label}
            </Badge>
          </div>
          <p className="text-xs text-muted-foreground">v{worker.version}</p>
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
