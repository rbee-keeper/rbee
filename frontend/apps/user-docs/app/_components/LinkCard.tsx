import { Card, CardDescription, CardHeader, CardTitle } from '@rbee/ui/atoms'
import { ArrowRight } from 'lucide-react'
import Link from 'next/link'
import type React from 'react'

interface LinkCardProps {
  title: string
  description: string
  href: string
  icon?: React.ReactNode
}

export function LinkCard({ title, description, href, icon }: LinkCardProps) {
  return (
    <Link href={href} className="group">
      <Card className="h-full transition-all hover:border-primary hover:shadow-md">
        <CardHeader>
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              {icon && <div className="text-primary">{icon}</div>}
              <CardTitle className="group-hover:text-primary transition-colors">{title}</CardTitle>
            </div>
            <ArrowRight className="h-5 w-5 text-muted-foreground group-hover:text-primary group-hover:translate-x-1 transition-all" />
          </div>
          <CardDescription>{description}</CardDescription>
        </CardHeader>
      </Card>
    </Link>
  )
}
