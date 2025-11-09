// TEAM-295: Navigation sidebar for Bee Keeper app
// Based on AppSidebar from queen-rbee UI
// TEAM-339: Simplified to work with react-resizable-panels (removed fixed-width Sidebar wrapper)
// TEAM-340: Added Queen navigation item with iframe page
// TEAM-342: Added Hives section with dynamic navigation to installed hives
// TEAM-405: Added Marketplace section with LLM Models, Image Models, Rbee Workers
// TEAM-413: Added DownloadPanel for tracking model/worker downloads
// TEAM-423: Renamed LLM Models → HuggingFace Models, Image Models → Civitai Models for clarity

import { ThemeToggle } from '@rbee/ui/molecules'
import {
  BrainIcon,
  CrownIcon,
  HelpCircleIcon,
  HomeIcon,
  ImageIcon,
  PackageIcon,
  ServerIcon,
  SettingsIcon,
} from 'lucide-react'
import { Link, useLocation } from 'react-router-dom'
import { useDownloadStore } from '@/store/downloadStore'
import type { SshHive } from '@/store/hiveQueries'
import { useInstalledHives, useSshHives } from '@/store/hiveQueries'
import { DownloadPanel } from './DownloadPanel'

export function KeeperSidebar() {
  const location = useLocation()
  const { data: hives = [] } = useSshHives()
  const { data: installedHives = [] } = useInstalledHives()

  // TEAM-413: Download tracking
  const { downloads, cancelDownload, retryDownload, removeDownload } = useDownloadStore()

  // TEAM-367: Filter hives by actual install status from backend
  const installedHivesList = hives.filter((hive: SshHive) => installedHives.includes(hive.host))

  const mainNavigation = [
    {
      title: 'Services',
      href: '/',
      icon: HomeIcon,
      tooltip: 'Manage services',
    },
    {
      title: 'Queen',
      href: '/queen',
      icon: CrownIcon,
      tooltip: 'Queen web interface',
    },
  ]

  const secondaryNavigation = [
    {
      title: 'Settings',
      href: '/settings',
      icon: SettingsIcon,
      tooltip: 'Configuration',
    },
    {
      title: 'Help',
      href: '/help',
      icon: HelpCircleIcon,
      tooltip: 'Documentation',
    },
  ]

  return (
    <div className="h-full w-full flex flex-col border-r border-border bg-background">
      {/* Main navigation */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-4 space-y-6">
          {/* Main section */}
          <div className="space-y-2">
            <h3 className="text-xs font-semibold font-serif text-muted-foreground uppercase tracking-wider px-2">
              Main
            </h3>
            <nav className="space-y-1">
              {mainNavigation.map((item) => {
                const isActive = location.pathname === item.href
                return (
                  <Link
                    key={item.href}
                    to={item.href}
                    title={item.tooltip}
                    className={`flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                      isActive ? 'bg-primary text-primary-foreground' : 'text-foreground hover:bg-muted'
                    }`}
                  >
                    <item.icon className="w-4 h-4 flex-shrink-0" />
                    <span className="truncate">{item.title}</span>
                  </Link>
                )
              })}
            </nav>
          </div>

          {/* TEAM-405: Marketplace section */}
          <div className="space-y-2">
            <h3 className="text-xs font-semibold font-serif text-muted-foreground uppercase tracking-wider px-2">
              Marketplace
            </h3>
            <nav className="space-y-1">
              <Link
                to="/marketplace/huggingface"
                title="Browse language models from HuggingFace"
                className={`flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  location.pathname === '/marketplace/huggingface'
                    ? 'bg-primary text-primary-foreground'
                    : 'text-foreground hover:bg-muted'
                }`}
              >
                <BrainIcon className="w-4 h-4 flex-shrink-0" />
                <span className="truncate">HuggingFace Models</span>
              </Link>
              <Link
                to="/marketplace/civitai"
                title="Browse image models from Civitai"
                className={`flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  location.pathname === '/marketplace/civitai'
                    ? 'bg-primary text-primary-foreground'
                    : 'text-foreground hover:bg-muted'
                }`}
              >
                <ImageIcon className="w-4 h-4 flex-shrink-0" />
                <span className="truncate">Civitai Models</span>
              </Link>
              <Link
                to="/marketplace/rbee-workers"
                title="Browse rbee worker binaries"
                className={`flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  location.pathname === '/marketplace/rbee-workers'
                    ? 'bg-primary text-primary-foreground'
                    : 'text-foreground hover:bg-muted'
                }`}
              >
                <PackageIcon className="w-4 h-4 flex-shrink-0" />
                <span className="truncate">Rbee Workers</span>
              </Link>
            </nav>
          </div>

          {/* TEAM-342: Hives section */}
          <div className="space-y-2">
            <h3 className="text-xs font-semibold font-serif text-muted-foreground uppercase tracking-wider px-2">
              Hives
            </h3>
            {installedHivesList.length === 0 ? (
              <p className="text-xs text-muted-foreground px-2 py-2">No hives installed yet</p>
            ) : (
              <nav className="space-y-1">
                {installedHivesList.map((hive) => {
                  const href = `/hive/${hive.host}`
                  const isActive = location.pathname === href
                  return (
                    <Link
                      key={hive.host}
                      to={href}
                      title={`${hive.user}@${hive.hostname}:${hive.port}`}
                      className={`flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                        isActive ? 'bg-primary text-primary-foreground' : 'text-foreground hover:bg-muted'
                      }`}
                    >
                      <ServerIcon className="w-4 h-4 flex-shrink-0" />
                      <span className="truncate">{hive.host}</span>
                    </Link>
                  )
                })}
              </nav>
            )}
          </div>
        </div>
      </div>

      {/* TEAM-413: Downloads section */}
      {downloads.length > 0 && (
        <DownloadPanel
          downloads={downloads}
          onCancel={cancelDownload}
          onRetry={retryDownload}
          onClear={removeDownload}
        />
      )}

      {/* System section */}
      <div className="border-t border-border p-4">
        <div className="space-y-2">
          <h3 className="text-xs font-semibold font-serif text-muted-foreground uppercase tracking-wider px-2">
            System
          </h3>
          <nav className="space-y-1">
            {secondaryNavigation.map((item) => {
              const isActive = location.pathname === item.href
              return (
                <Link
                  key={item.href}
                  to={item.href}
                  title={item.tooltip}
                  className={`flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                    isActive ? 'bg-primary text-primary-foreground' : 'text-foreground hover:bg-muted'
                  }`}
                >
                  <item.icon className="w-4 h-4 flex-shrink-0" />
                  <span className="truncate">{item.title}</span>
                </Link>
              )
            })}
          </nav>
        </div>
      </div>

      {/* Footer */}
      <div className="border-t border-border p-4">
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground font-mono">v0.1.0</span>
          <ThemeToggle />
        </div>
      </div>
    </div>
  )
}
