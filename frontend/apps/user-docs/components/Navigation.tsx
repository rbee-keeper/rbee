// TEAM-424: User docs navigation - uses config-driven Navigation component
import { Navigation as BaseNavigation } from '@rbee/ui/organisms/Navigation'
import { userDocsNavConfig } from '@/config/navigationConfig'

export function Navigation() {
  return <BaseNavigation config={userDocsNavConfig} />
}
