// TEAM-502: Language Filter Component
// Shows available languages as checkboxes

import React, { useState } from 'react'
import { ChevronDown, ChevronUp, Globe } from 'lucide-react'
import { Input } from '@rbee/ui/atoms/Input'
import { Checkbox } from '@rbee/ui/atoms/Checkbox'
import { Label } from '@rbee/ui/atoms/Label'
import { Button } from '@rbee/ui/atoms/Button'

interface LanguageFilterProps {
  languages: string[]
  selectedLanguages: string[]
  onLanguagesChange: (languages: string[]) => void
}

/**
 * Language checkbox list component
 */
export const LanguageFilter: React.FC<LanguageFilterProps> = ({
  languages,
  selectedLanguages,
  onLanguagesChange
}) => {
  const [showAll, setShowAll] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  
  // Common languages to show by default
  const commonLanguages = [
    'en',
    'zh',
    'es',
    'fr',
    'de',
    'ja',
    'ko',
    'ru',
    'pt',
    'it',
    'ar',
    'hi',
    'multilingual'
  ]
  
  // Language display names
  const languageNames: Record<string, string> = {
    'en': 'English',
    'zh': 'Chinese',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ru': 'Russian',
    'pt': 'Portuguese',
    'it': 'Italian',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'tr': 'Turkish',
    'pl': 'Polish',
    'nl': 'Dutch',
    'sv': 'Swedish',
    'da': 'Danish',
    'no': 'Norwegian',
    'fi': 'Finnish',
    'he': 'Hebrew',
    'th': 'Thai',
    'vi': 'Vietnamese',
    'id': 'Indonesian',
    'ms': 'Malay',
    'tl': 'Filipino',
    'ur': 'Urdu',
    'bn': 'Bengali',
    'ta': 'Tamil',
    'te': 'Telugu',
    'mr': 'Marathi',
    'gu': 'Gujarati',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'pa': 'Punjabi',
    'multilingual': 'Multilingual'
  }
  
  // Language flags/icons
  const languageFlags: Record<string, string> = {
    'en': '🇺🇸',
    'zh': '🇨🇳',
    'es': '🇪🇸',
    'fr': '🇫🇷',
    'de': '🇩🇪',
    'ja': '🇯🇵',
    'ko': '🇰🇷',
    'ru': '🇷🇺',
    'pt': '🇵🇹',
    'it': '🇮🇹',
    'ar': '🇸🇦',
    'hi': '🇮🇳',
    'tr': '🇹🇷',
    'pl': '🇵🇱',
    'nl': '🇳🇱',
    'sv': '🇸🇪',
    'da': '🇩🇰',
    'no': '🇳🇴',
    'fi': '🇫🇮',
    'he': '🇮🇱',
    'th': '🇹🇭',
    'vi': '🇻🇳',
    'id': '🇮🇩',
    'ms': '🇲🇾',
    'tl': '🇵🇭',
    'ur': '🇵🇰',
    'bn': '🇧🇩',
    'ta': '🇱🇰',
    'te': '🇮🇳',
    'mr': '🇮🇳',
    'gu': '🇮🇳',
    'kn': '🇮🇳',
    'ml': '🇮🇳',
    'pa': '🇮🇳',
    'multilingual': '🌍'
  }
  
  // Filter languages by search
  const filteredLanguages = languages.filter(lang => {
    const query = searchQuery.toLowerCase()
    const name = languageNames[lang]?.toLowerCase() || lang.toLowerCase()
    return name.includes(query) || lang.includes(query)
  })
  
  // Sort: common languages first, then alphabetical
  const sortedLanguages = [...filteredLanguages].sort((a, b) => {
    const aIsCommon = commonLanguages.includes(a)
    const bIsCommon = commonLanguages.includes(b)
    
    if (aIsCommon && !bIsCommon) return -1
    if (!aIsCommon && bIsCommon) return 1
    if (aIsCommon && bIsCommon) return commonLanguages.indexOf(a) - commonLanguages.indexOf(b)
    
    return (languageNames[a] || a).localeCompare(languageNames[b] || b)
  })
  
  // Show first 12 languages by default
  const displayedLanguages = showAll ? sortedLanguages : sortedLanguages.slice(0, 12)
  const hasMore = sortedLanguages.length > 12
  
  const handleLanguageToggle = (language: string) => {
    if (selectedLanguages.includes(language)) {
      // Remove language
      onLanguagesChange(selectedLanguages.filter(l => l !== language))
    } else {
      // Add language
      onLanguagesChange([...selectedLanguages, language])
    }
  }
  
  const handleSelectCommon = () => {
    // Select common languages
    const common = commonLanguages.filter(lang => languages.includes(lang))
    onLanguagesChange(common)
  }
  
  const handleSelectMultilingual = () => {
    // Select only multilingual models
    onLanguagesChange(['multilingual'])
  }

  return (
    <div className="space-y-3">
      {/* Quick Actions */}
      <div className="flex gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={handleSelectCommon}
          className="text-xs"
        >
          Common Languages
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={handleSelectMultilingual}
          className="text-xs"
        >
          Multilingual Only
        </Button>
      </div>
      
      {/* Search */}
      <div className="relative">
        <Globe className="absolute left-3 top-1/2 transform -translate-y-1/2 w-3 h-3 text-gray-400" />
        <Input
          type="text"
          placeholder="Search languages..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="h-8 pl-8 text-sm"
        />
      </div>
      
      {/* Language List */}
      <div className="space-y-2">
        {displayedLanguages.map((language) => {
          const isSelected = selectedLanguages.includes(language)
          const displayName = languageNames[language] || language
          const flag = languageFlags[language] || '🏳️'
          
          return (
            <div
              key={language}
              className={`
                flex items-center gap-3 p-2 rounded-lg transition-all
                ${isSelected 
                  ? 'bg-blue-50 border border-blue-200 hover:bg-blue-100' 
                  : 'bg-gray-50 border border-gray-200 hover:bg-gray-100'
                }
              `}
            >
              <Checkbox
                id={`language-${language}`}
                checked={isSelected}
                onCheckedChange={() => handleLanguageToggle(language)}
              />
              <Label
                htmlFor={`language-${language}`}
                className="flex items-center gap-2 flex-1 cursor-pointer"
              >
                <span className="text-base">{flag}</span>
                <span className="font-medium text-gray-900 text-sm">
                  {displayName}
                </span>
                <span className="text-xs text-gray-500">
                  ({language})
                </span>
              </Label>
            </div>
          )
        })}
        
        {hasMore && (
          <button
            onClick={() => setShowAll(!showAll)}
            className="flex items-center gap-2 text-sm text-blue-600 hover:text-blue-700 transition-colors"
          >
            {showAll ? (
              <>
                <ChevronUp className="w-4 h-4" />
                Show less
              </>
            ) : (
              <>
                <ChevronDown className="w-4 h-4" />
                Show {sortedLanguages.length - 12} more languages
              </>
            )}
          </button>
        )}
        
        {filteredLanguages.length === 0 && (
          <div className="text-center py-4 text-gray-500 text-sm">
            No languages found matching "{searchQuery}"
          </div>
        )}
        
        {languages.length === 0 && (
          <div className="text-center py-4 text-gray-500 text-sm">
            No languages available. Select a worker first.
          </div>
        )}
      </div>
    </div>
  )
}

export default LanguageFilter
