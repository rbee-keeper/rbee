// TEAM-502: License Filter Component
// Shows available licenses as checkboxes

import React, { useState } from 'react'
import { ChevronDown, ChevronUp, Shield, AlertTriangle } from 'lucide-react'
import { Input } from '@rbee/ui/atoms/Input'
import { Checkbox } from '@rbee/ui/atoms/Checkbox'
import { Label } from '@rbee/ui/atoms/Label'
import { Button } from '@rbee/ui/atoms/Button'

interface LicenseFilterProps {
  licenses: string[]
  selectedLicenses: string[]
  onLicensesChange: (licenses: string[]) => void
}

/**
 * License checkbox list component
 */
export const LicenseFilter: React.FC<LicenseFilterProps> = ({
  licenses,
  selectedLicenses,
  onLicensesChange
}) => {
  const [showAll, setShowAll] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  
  // License categories with descriptions
  const licenseCategories = {
    'permissive': {
      name: 'Permissive',
      description: 'Minimal restrictions, commercial friendly',
      licenses: ['mit', 'apache-2.0', 'bsd-3-clause', 'bsd-2-clause', 'isc', 'unlicense']
    },
    'copyleft': {
      name: 'Copyleft',
      description: 'Share-alike requirements, source disclosure',
      licenses: ['gpl-3.0', 'gpl-2.0', 'lgpl-3.0', 'lgpl-2.1', 'agpl-3.0']
    },
    'creative': {
      name: 'Creative Commons',
      description: 'Content and media licenses',
      licenses: ['cc-by-4.0', 'cc-by-sa-4.0', 'cc-by-nc-4.0', 'cc-by-nd-4.0', 'cc0-1.0']
    },
    'ai': {
      name: 'AI-Specific',
      description: 'Special licenses for AI models',
      licenses: ['llama2', 'llama3', 'llama3.1', 'llama3.2', 'gemma', 'openrail', 'bigscience-openrail-m', 'creativeml-openrail-m']
    },
    'other': {
      name: 'Other',
      description: 'Other specialized licenses',
      licenses: ['afl-3.0', 'artistic-2.0', 'epl-2.0', 'mpl-2.0', 'odc-by', 'odbl']
    }
  }
  
  // License display names and descriptions
  const licenseInfo: Record<string, { name: string; description: string; category: string; risk: 'low' | 'medium' | 'high' }> = {
    'mit': { name: 'MIT License', description: 'Permissive, simple, commercial friendly', category: 'permissive', risk: 'low' },
    'apache-2.0': { name: 'Apache 2.0', description: 'Permissive, patent protection', category: 'permissive', risk: 'low' },
    'bsd-3-clause': { name: 'BSD 3-Clause', description: 'Permissive, no endorsement clause', category: 'permissive', risk: 'low' },
    'bsd-2-clause': { name: 'BSD 2-Clause', description: 'Permissive, very simple', category: 'permissive', risk: 'low' },
    'isc': { name: 'ISC License', description: 'Permissive, modern equivalent of BSD', category: 'permissive', risk: 'low' },
    'unlicense': { name: 'Unlicense', description: 'Public domain, no restrictions', category: 'permissive', risk: 'low' },
    
    'gpl-3.0': { name: 'GPL 3.0', description: 'Strong copyleft, source disclosure required', category: 'copyleft', risk: 'high' },
    'gpl-2.0': { name: 'GPL 2.0', description: 'Strong copyleft, source disclosure required', category: 'copyleft', risk: 'high' },
    'lgpl-3.0': { name: 'LGPL 3.0', description: 'Weak copyleft, library linking allowed', category: 'copyleft', risk: 'medium' },
    'lgpl-2.1': { name: 'LGPL 2.1', description: 'Weak copyleft, library linking allowed', category: 'copyleft', risk: 'medium' },
    'agpl-3.0': { name: 'AGPL 3.0', description: 'Strong copyleft, network use requires source', category: 'copyleft', risk: 'high' },
    
    'cc-by-4.0': { name: 'CC BY 4.0', description: 'Attribution required, commercial allowed', category: 'creative', risk: 'low' },
    'cc-by-sa-4.0': { name: 'CC BY-SA 4.0', description: 'Attribution + Share-Alike required', category: 'creative', risk: 'medium' },
    'cc-by-nc-4.0': { name: 'CC BY-NC 4.0', description: 'Non-commercial only, attribution required', category: 'creative', risk: 'medium' },
    'cc-by-nd-4.0': { name: 'CC BY-ND 4.0', description: 'No derivatives, attribution required', category: 'creative', risk: 'medium' },
    'cc0-1.0': { name: 'CC0 1.0', description: 'Public domain dedication', category: 'creative', risk: 'low' },
    
    'llama2': { name: 'Llama 2 License', description: 'Custom license, commercial restrictions', category: 'ai', risk: 'high' },
    'llama3': { name: 'Llama 3 License', description: 'Custom license, more permissive than Llama 2', category: 'ai', risk: 'medium' },
    'llama3.1': { name: 'Llama 3.1 License', description: 'Custom license, allows commercial use', category: 'ai', risk: 'medium' },
    'llama3.2': { name: 'Llama 3.2 License', description: 'Custom license, multimodal models', category: 'ai', risk: 'medium' },
    'gemma': { name: 'Gemma License', description: 'Google custom license, usage restrictions', category: 'ai', risk: 'medium' },
    'openrail': { name: 'OpenRAIL', description: 'Responsible AI license, use restrictions', category: 'ai', risk: 'medium' },
    'bigscience-openrail-m': { name: 'BigScience OpenRAIL-M', description: 'Responsible AI license for BLOOM', category: 'ai', risk: 'medium' },
    'creativeml-openrail-m': { name: 'CreativeML OpenRAIL-M', description: 'Responsible AI license for Stable Diffusion', category: 'ai', risk: 'medium' }
  }
  
  // Filter licenses by search
  const filteredLicenses = licenses.filter(license => {
    const query = searchQuery.toLowerCase()
    const info = licenseInfo[license]
    if (!info) return license.toLowerCase().includes(query)
    
    return (
      license.toLowerCase().includes(query) ||
      info.name.toLowerCase().includes(query) ||
      info.description.toLowerCase().includes(query) ||
      info.category.toLowerCase().includes(query)
    )
  })
  
  // Group licenses by category
  const groupedLicenses: Record<string, string[]> = {}
  filteredLicenses.forEach(license => {
    const info = licenseInfo[license]
    const category = info?.category || 'other'
    if (!groupedLicenses[category]) {
      groupedLicenses[category] = []
    }
    groupedLicenses[category].push(license)
  })
  
  // Sort categories: permissive first, then others
  const sortedCategories = Object.keys(groupedLicenses).sort((a, b) => {
    if (a === 'permissive') return -1
    if (b === 'permissive') return 1
    if (a === 'ai') return -1
    if (b === 'ai') return 1
    return a.localeCompare(b)
  })
  
  const handleLicenseToggle = (license: string) => {
    if (selectedLicenses.includes(license)) {
      // Remove license
      onLicensesChange(selectedLicenses.filter(l => l !== license))
    } else {
      // Add license
      onLicensesChange([...selectedLicenses, license])
    }
  }
  
  const handleSelectCategory = (category: string) => {
    const categoryLicenses = groupedLicenses[category] || []
    const newSelection = [...new Set([...selectedLicenses, ...categoryLicenses])]
    onLicensesChange(newSelection)
  }
  
  const handleSelectLowRisk = () => {
    const lowRiskLicenses = Object.entries(licenseInfo)
      .filter(([, info]) => info.risk === 'low')
      .map(([license]) => license)
      .filter((license) => licenses.includes(license))

    onLicensesChange(lowRiskLicenses)
  }
  
  const getRiskIcon = (risk: 'low' | 'medium' | 'high') => {
    switch (risk) {
      case 'low': return <Shield className="w-3 h-3 text-sidebar-foreground" />
      case 'medium': return <AlertTriangle className="w-3 h-3 text-sidebar-foreground" />
      case 'high': return <AlertTriangle className="w-3 h-3 text-sidebar-foreground" />
    }
  }

  return (
    <div className="space-y-3">
      {/* Quick Actions */}
      <div className="flex flex-wrap gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={handleSelectLowRisk}
          className="text-xs bg-sidebar-accent/10 text-sidebar-foreground border-sidebar-border hover:bg-sidebar-accent/20"
        >
          Low Risk Only
        </Button>
        {sortedCategories.map(category => (
          <Button
            key={category}
            variant="outline"
            size="sm"
            onClick={() => handleSelectCategory(category)}
            className="text-xs capitalize"
          >
            {licenseCategories[category as keyof typeof licenseCategories]?.name || category}
          </Button>
        ))}
      </div>
      
      {/* Search */}
      <div className="relative">
        <Shield className="absolute left-3 top-1/2 transform -translate-y-1/2 w-3 h-3 text-muted-foreground" />
        <Input
          type="text"
          placeholder="Search licenses..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="h-8 pl-8 text-sm"
        />
      </div>
      
      {/* License List by Category */}
      <div className="space-y-4">
        {sortedCategories.map(category => {
          const categoryInfo = licenseCategories[category as keyof typeof licenseCategories]
          const categoryLicenses = groupedLicenses[category]
          
          if (!categoryLicenses || categoryLicenses.length === 0) return null
          
          return (
            <div key={category}>
              <div className="flex items-center gap-2 mb-2">
                <h4 className="text-sm font-medium text-sidebar-foreground capitalize">
                  {categoryInfo?.name || category}
                </h4>
                <span className="text-xs text-muted-foreground">
                  ({categoryLicenses.length})
                </span>
              </div>
              
              {categoryInfo?.description && (
                <p className="text-xs text-muted-foreground mb-2">
                  {categoryInfo.description}
                </p>
              )}
              
              <div className="space-y-2">
                {categoryLicenses.map(license => {
                  const isSelected = selectedLicenses.includes(license)
                  const info = licenseInfo[license]
                  
                  return (
                    <div
                      key={license}
                      className={`
                        flex items-start gap-3 p-2 rounded-lg transition-all
                        ${isSelected 
                          ? 'bg-sidebar-accent/10 border border-sidebar-accent hover:bg-sidebar-accent/20' 
                          : 'bg-muted border border-sidebar-border hover:bg-muted/80'
                        }
                      `}
                    >
                      <Checkbox
                        id={`license-${license}`}
                        checked={isSelected}
                        onCheckedChange={() => handleLicenseToggle(license)}
                      />
                      <Label
                        htmlFor={`license-${license}`}
                        className="flex-1 min-w-0 cursor-pointer"
                      >
                        <div className="flex items-center gap-2">
                          {info && getRiskIcon(info.risk)}
                          <span className="font-medium text-sidebar-foreground text-sm">
                            {info?.name || license}
                          </span>
                        </div>
                        <p className="text-xs text-muted-foreground mt-0.5">
                          {info?.description || 'License for model usage'}
                        </p>
                        <p className="text-xs text-muted-foreground mt-0.5">
                          {license}
                        </p>
                      </Label>
                    </div>
                  )
                })}
              </div>
            </div>
          )
        })}
        
        {filteredLicenses.length === 0 && (
          <div className="text-center py-4 text-muted-foreground text-sm">
            No licenses found matching "{searchQuery}"
          </div>
        )}
        
        {licenses.length === 0 && (
          <div className="text-center py-4 text-muted-foreground text-sm">
            No licenses available. Select a worker first.
          </div>
        )}
      </div>
    </div>
  )
}

export default LicenseFilter
