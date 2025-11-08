import { useMDXComponents as getDocsComponents } from 'nextra-theme-docs'

// Import existing components from @rbee/ui
import { 
  Alert, AlertTitle, AlertDescription,
  Accordion, AccordionItem, AccordionTrigger, AccordionContent,
  Tabs, TabsList, TabsTrigger, TabsContent,
  Badge, Button,
  Table, TableHeader, TableBody, TableRow, TableCell, TableHead,
  Input,
  Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter,
  CodeSnippet,
  BrandMark, BrandWordmark,
  Separator,
} from '@rbee/ui/atoms'

import { 
  CodeBlock,
  TerminalWindow,
  ThemeToggle,
} from '@rbee/ui/molecules'

// Import custom MDX wrappers
import { Callout } from './app/docs/_components/Callout'
import { LinkCard } from './app/docs/_components/LinkCard'
import { CardGrid } from './app/docs/_components/CardGrid'
import { CodeTabs } from './app/docs/_components/CodeTabs'
import { APIParameterTable } from './app/docs/_components/APIParameterTable'

export function useMDXComponents(components: any): any {
  return {
    ...getDocsComponents(components),
    
    // Alerts
    Alert, AlertTitle, AlertDescription,
    
    // Accordion
    Accordion, AccordionItem, AccordionTrigger, AccordionContent,
    
    // Tabs
    Tabs, TabsList, TabsTrigger, TabsContent,
    
    // Code
    CodeSnippet,
    CodeBlock,
    TerminalWindow,
    
    // Tables
    Table, TableHeader, TableBody, TableRow, TableCell, TableHead,
    
    // Cards
    Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter,
    
    // UI Elements
    Badge, Button, Input, Separator,
    
    // Branding
    BrandMark, BrandWordmark,
    
    // Theme
    ThemeToggle,
    
    // Custom MDX wrappers
    Callout,
    LinkCard,
    CardGrid,
    CodeTabs,
    APIParameterTable,
    
    ...components,
  }
}
