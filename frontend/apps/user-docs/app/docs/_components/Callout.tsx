import { Alert, AlertTitle, AlertDescription } from '@rbee/ui/atoms'
import { Info, AlertTriangle, XCircle, CheckCircle } from 'lucide-react'

interface CalloutProps {
  variant?: 'info' | 'warning' | 'error' | 'success'
  title?: string
  children: React.ReactNode
}

const icons = {
  info: Info,
  warning: AlertTriangle,
  error: XCircle,
  success: CheckCircle,
}

const variantMap = {
  info: 'default',
  warning: 'default',
  error: 'destructive',
  success: 'default',
} as const

export function Callout({ 
  variant = 'info', 
  title, 
  children 
}: CalloutProps) {
  const Icon = icons[variant]
  
  return (
    <Alert variant={variantMap[variant]} className="my-6">
      <Icon className="h-4 w-4" />
      {title && <AlertTitle>{title}</AlertTitle>}
      <AlertDescription>{children}</AlertDescription>
    </Alert>
  )
}
