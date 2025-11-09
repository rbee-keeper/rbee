// TEAM-463: Direct link component for navbar (no dropdown)
import Link from 'next/link'
import type { DirectLinkSection } from '@rbee/ui/organisms/Navigation/types'

export interface DirectLinkProps extends Omit<DirectLinkSection, 'type'> {}

export function DirectLink({ label, href }: DirectLinkProps) {
  return (
    <Link
      href={href}
      className="text-sm font-medium text-foreground/80 hover:text-foreground transition-colors px-2"
    >
      {label}
    </Link>
  )
}
