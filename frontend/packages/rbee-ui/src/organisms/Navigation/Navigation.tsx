'use client'

// TEAM-460: Navigation organism - config-driven, uses molecules
// Replaces hardcoded navigation with NavigationDropdown, LinkGroup, NavigationActions

import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@rbee/ui/atoms/Accordion'
import { IconButton } from '@rbee/ui/atoms/IconButton'
import { NavigationMenu, NavigationMenuList } from '@rbee/ui/atoms/NavigationMenu'
import { Separator } from '@rbee/ui/atoms/Separator'
import { Sheet, SheetContent, SheetTitle, SheetTrigger } from '@rbee/ui/atoms/Sheet'
import { BrandLogo, NavigationDropdown, LinkGroup, NavigationActions, NavLink } from '@rbee/ui/molecules'
import { Menu, X } from 'lucide-react'
import { usePathname } from 'next/navigation'
import { useState } from 'react'
import type { NavigationConfig } from './types'

export interface NavigationProps {
  config: NavigationConfig
}

export function Navigation({ config }: NavigationProps) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const pathname = usePathname()

  return (
    <>
      {/* Skip to content link */}
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-[60] rounded-md bg-primary px-3 py-2 text-primary-foreground shadow"
      >
        Skip to content
      </a>

      <nav
        aria-label="Primary"
        className="fixed top-0 inset-x-0 z-50 bg-background/95 supports-[backdrop-filter]:bg-background/70 backdrop-blur-sm border-b border-border/60"
      >
        <div className="relative before:absolute before:inset-x-0 before:top-0 before:h-px before:bg-gradient-to-r before:from-transparent before:via-primary/15 before:to-transparent">
          <div className="px-4 sm:px-6 lg:px-8 mx-auto max-w-7xl">
            <div className="grid grid-cols-[auto_1fr_auto] items-center h-16 md:h-14">
              {/* Zone A: Logo + Brand */}
              <BrandLogo priority />

              {/* Zone B: Navigation (Desktop) - Config-driven */}
              <div className="hidden md:flex items-center justify-center gap-6 font-sans">
                {config.sections.some((s) => s.type === 'dropdown') ? (
                  <NavigationMenu viewport={false}>
                    <NavigationMenuList className="gap-2">
                      {config.sections.map((section, index) => {
                        if (section.type === 'dropdown') {
                          return (
                            <NavigationDropdown
                              key={index}
                              title={section.title}
                              links={section.links}
                              cta={section.cta}
                              width={section.width}
                            />
                          )
                        }
                        return null
                      })}
                    </NavigationMenuList>
                  </NavigationMenu>
                ) : (
                  <>
                    {config.sections.map((section, index) => {
                      if (section.type === 'linkGroup') {
                        return <LinkGroup key={index} links={section.links} />
                      }
                      if (section.type === 'separator') {
                        return <div key={index} className="h-4 w-px bg-border" />
                      }
                      return null
                    })}
                  </>
                )}
              </div>

              {/* Zone C: Actions (Desktop) */}
              <NavigationActions
                docs={config.actions.docs}
                github={config.actions.github}
                cta={config.actions.cta}
              />

              {/* Mobile Menu Toggle */}
              <div className="md:hidden justify-self-end">
                <Sheet open={mobileMenuOpen} onOpenChange={setMobileMenuOpen}>
                  <SheetTrigger asChild>
                    <IconButton
                      aria-label="Open navigation menu"
                      className="hover:bg-accent"
                    >
                      <Menu className="size-5" />
                    </IconButton>
                  </SheetTrigger>
                  <SheetContent
                    side="right"
                    className="w-[min(400px,100vw)] sm:w-[400px] p-0 overflow-y-auto"
                  >
                    <div className="flex flex-col h-full">
                      {/* Mobile Header */}
                      <div className="flex items-center justify-between p-4 border-b border-border">
                        <SheetTitle className="text-lg font-semibold">Menu</SheetTitle>
                        <IconButton
                          onClick={() => setMobileMenuOpen(false)}
                          aria-label="Close menu"
                          className="hover:bg-accent"
                        >
                          <X className="size-5" />
                        </IconButton>
                      </div>

                      {/* Mobile Navigation */}
                      <div className="flex-1 overflow-y-auto p-4">
                        <Accordion type="multiple" className="w-full">
                          {config.sections.map((section, index) => {
                            if (section.type === 'dropdown') {
                              return (
                                <AccordionItem key={index} value={`section-${index}`}>
                                  <AccordionTrigger className="text-sm font-medium hover:no-underline">
                                    {section.title}
                                  </AccordionTrigger>
                                  <AccordionContent>
                                    <div className="flex flex-col gap-2 pt-2">
                                      {section.links.map((link) => (
                                        <NavLink
                                          key={link.href}
                                          href={link.href}
                                          className="block rounded-md p-2 text-sm hover:bg-accent"
                                          onClick={() => setMobileMenuOpen(false)}
                                        >
                                          {link.label}
                                        </NavLink>
                                      ))}
                                    </div>
                                  </AccordionContent>
                                </AccordionItem>
                              )
                            }

                            if (section.type === 'linkGroup') {
                              return (
                                <div key={index} className="py-2">
                                  {section.links.map((link) => (
                                    <NavLink
                                      key={link.href}
                                      href={link.href}
                                      className="block rounded-md p-2 text-sm hover:bg-accent"
                                      onClick={() => setMobileMenuOpen(false)}
                                    >
                                      {link.label}
                                      {link.badge && (
                                        <span className="ml-2 text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
                                          {link.badge}
                                        </span>
                                      )}
                                    </NavLink>
                                  ))}
                                </div>
                              )
                            }

                            if (section.type === 'separator') {
                              return <Separator key={index} className="my-2" />
                            }

                            return null
                          })}
                        </Accordion>
                      </div>
                    </div>
                  </SheetContent>
                </Sheet>
              </div>
            </div>
          </div>
        </div>
      </nav>
    </>
  )
}
