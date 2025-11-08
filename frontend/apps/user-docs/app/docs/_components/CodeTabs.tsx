'use client'

import { useState, useEffect } from 'react'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@rbee/ui/atoms'
import { CodeBlock } from '@rbee/ui/molecules'

interface CodeTab {
  label: string
  language: 'bash' | 'python' | 'javascript' | 'typescript' | 'json' | 'toml' | 'rust'
  code: string
  title?: string
}

interface CodeTabsProps {
  tabs: CodeTab[]
  defaultTab?: string
  storageKey?: string
}

export function CodeTabs({ 
  tabs, 
  defaultTab, 
  storageKey = 'code-tabs-preference' 
}: CodeTabsProps) {
  const [activeTab, setActiveTab] = useState(
    defaultTab || tabs[0]?.label || ''
  )
  
  useEffect(() => {
    // Load saved preference from localStorage
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem(storageKey)
      if (saved && tabs.find(t => t.label === saved)) {
        setActiveTab(saved)
      }
    }
  }, [storageKey, tabs])
  
  const handleTabChange = (value: string) => {
    setActiveTab(value)
    if (typeof window !== 'undefined') {
      localStorage.setItem(storageKey, value)
    }
  }
  
  return (
    <Tabs value={activeTab} onValueChange={handleTabChange} className="my-6">
      <TabsList>
        {tabs.map(tab => (
          <TabsTrigger key={tab.label} value={tab.label}>
            {tab.label}
          </TabsTrigger>
        ))}
      </TabsList>
      {tabs.map(tab => (
        <TabsContent key={tab.label} value={tab.label} className="mt-0">
          <CodeBlock 
            code={tab.code}
            language={tab.language}
            title={tab.title}
            copyable={true}
          />
        </TabsContent>
      ))}
    </Tabs>
  )
}
