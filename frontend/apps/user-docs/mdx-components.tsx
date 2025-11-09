// Import existing components from @rbee/ui
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
  Alert,
  AlertDescription,
  AlertTitle,
  Badge,
  BrandMark,
  BrandWordmark,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
  CodeSnippet,
  Input,
  Separator,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from '@rbee/ui/atoms'
import { CodeBlock, TerminalWindow, ThemeToggle } from '@rbee/ui/molecules'
import { useMDXComponents as getDocsComponents } from 'nextra-theme-docs'
import { APIParameterTable } from './app/docs/_components/APIParameterTable'
// Import custom MDX wrappers
import { Callout } from './app/docs/_components/Callout'
import { CardGrid } from './app/docs/_components/CardGrid'
import { CodeTabs } from './app/docs/_components/CodeTabs'
import { LinkCard } from './app/docs/_components/LinkCard'

export function useMDXComponents(components: any): any {
  return {
    ...getDocsComponents(components),

    // Alerts
    Alert,
    AlertTitle,
    AlertDescription,

    // Accordion
    Accordion,
    AccordionItem,
    AccordionTrigger,
    AccordionContent,

    // Tabs
    Tabs,
    TabsList,
    TabsTrigger,
    TabsContent,

    // Code
    CodeSnippet,
    CodeBlock,
    TerminalWindow,

    // Tables
    Table,
    TableHeader,
    TableBody,
    TableRow,
    TableCell,
    TableHead,

    // Cards
    Card,
    CardHeader,
    CardTitle,
    CardDescription,
    CardContent,
    CardFooter,

    // UI Elements
    Badge,
    Button,
    Input,
    Separator,

    // Branding
    BrandMark,
    BrandWordmark,

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
