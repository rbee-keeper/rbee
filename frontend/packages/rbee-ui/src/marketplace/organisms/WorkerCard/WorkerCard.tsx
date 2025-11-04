// TEAM-401: Marketplace worker card organism
// TEAM-404: Now uses canonical WorkerType from marketplace-sdk
import { Badge } from '@rbee/ui/atoms/Badge'
import { Button } from '@rbee/ui/atoms/Button'
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@rbee/ui/atoms/Card'
import { Cpu, Download } from 'lucide-react'
import * as React from 'react'

// TEAM-404: Import canonical WorkerType from marketplace-sdk
// This ensures type consistency across Rust (artifacts-contract), WASM (marketplace-sdk), and React
// TODO: Once marketplace-sdk is published to npm, import from '@rbee/marketplace-sdk'
// For now, we use the inline type that matches the generated WASM types
export type WorkerType = 'cpu' | 'cuda' | 'metal'

export interface WorkerCardProps {
  worker: {
    id: string
    name: string
    description: string
    version: string
    platform: string[]
    architecture: string[]
    workerType: WorkerType
  }
  onAction?: (workerId: string) => void
  actionButton?: React.ReactNode
}

const workerTypeConfig = {
  cpu: { label: 'CPU', variant: 'secondary' as const },
  cuda: { label: 'CUDA', variant: 'default' as const },
  metal: { label: 'Metal', variant: 'accent' as const },
}

export function WorkerCard({ worker, onAction, actionButton }: WorkerCardProps) {
  const typeConfig = workerTypeConfig[worker.workerType]

  return (
    <Card className="h-full flex flex-col hover:shadow-md transition-shadow">
      <CardHeader>
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <Cpu className="size-4 text-muted-foreground shrink-0" />
              <CardTitle className="truncate">{worker.name}</CardTitle>
            </div>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span>v{worker.version}</span>
              <span>•</span>
              <Badge variant={typeConfig.variant} className="text-[10px] px-1.5 py-0">
                {typeConfig.label}
              </Badge>
            </div>
          </div>
          {actionButton && <CardAction>{actionButton}</CardAction>}
        </div>
        <CardDescription className="line-clamp-2">{worker.description}</CardDescription>
      </CardHeader>

      <CardContent className="flex-1">
        <div className="space-y-3">
          <div>
            <div className="text-xs font-medium text-muted-foreground mb-1.5">Platforms</div>
            <div className="flex flex-wrap gap-1.5">
              {worker.platform.map((platform) => (
                <Badge key={platform} variant="outline">
                  {platform}
                </Badge>
              ))}
            </div>
          </div>

          <div>
            <div className="text-xs font-medium text-muted-foreground mb-1.5">Architecture</div>
            <div className="flex flex-wrap gap-1.5">
              {worker.architecture.map((arch) => (
                <Badge key={arch} variant="secondary">
                  {arch}
                </Badge>
              ))}
            </div>
          </div>
        </div>
      </CardContent>

      <CardFooter className="border-t pt-4">
        {!actionButton && onAction && (
          <Button size="sm" className="w-full" onClick={() => onAction(worker.id)}>
            <Download className="size-3.5" />
            Install Worker
          </Button>
        )}
      </CardFooter>
    </Card>
  )
}
