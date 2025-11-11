// TEAM-476: Marketplace navigation - uses config-driven Navigation component
import { Navigation } from '@rbee/ui/organisms/Navigation'
import { marketplaceNavConfig } from '@/config/navigationConfig'

export function MarketplaceNav() {
  return <Navigation config={marketplaceNavConfig} />
}
