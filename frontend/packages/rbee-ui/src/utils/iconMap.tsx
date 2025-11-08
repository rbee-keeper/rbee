// TEAM-XXX: Icon mapping utility for SSG-compatible icon props
// Maps string icon names to Lucide React components
// This allows passing icon names as strings in props instead of JSX elements

import {
  Activity,
  AlertCircle,
  AlertTriangle,
  ArrowRight,
  Banknote,
  BarChart,
  BookOpen,
  Brain,
  Briefcase,
  Bug,
  Building,
  Calendar,
  Check,
  CheckCircle,
  Clock,
  Cloud,
  Code,
  Cpu,
  Database,
  DollarSign,
  Eye,
  Factory,
  FileCheck,
  FileCode,
  FileSearch,
  FileText,
  FileX,
  FlaskConical,
  Gamepad,
  Gauge,
  GitBranch,
  GitPullRequest,
  Github,
  Globe,
  GraduationCap,
  Heart,
  Home,
  Home as HomeIcon,
  Image,
  KeyRound,
  Landmark,
  Laptop,
  Layers,
  Lock,
  MemoryStick,
  MessageSquare,
  Mic,
  Monitor,
  Network,
  RefreshCw,
  Rocket,
  Scale,
  Search,
  Server,
  Settings,
  Shield,
  ShieldAlert,
  ShieldCheck,
  Shuffle,
  Sliders,
  Sparkles,
  Star,
  Target,
  Terminal,
  TestTube,
  Timer,
  TrendingDown,
  TrendingUp,
  Users,
  Wifi,
  Workflow,
  Wrench,
  X,
  XCircle,
  Zap,
  type LucideIcon,
} from 'lucide-react'

export type IconName =
  | 'Activity'
  | 'AlertCircle'
  | 'AlertTriangle'
  | 'ArrowRight'
  | 'Banknote'
  | 'BarChart'
  | 'BookOpen'
  | 'Brain'
  | 'Briefcase'
  | 'Bug'
  | 'Building'
  | 'Calendar'
  | 'Check'
  | 'CheckCircle'
  | 'Clock'
  | 'Cloud'
  | 'Code'
  | 'Cpu'
  | 'Database'
  | 'DollarSign'
  | 'Eye'
  | 'Factory'
  | 'FileCheck'
  | 'FileCode'
  | 'FileSearch'
  | 'FileText'
  | 'FileX'
  | 'FlaskConical'
  | 'Gamepad'
  | 'Gauge'
  | 'GitBranch'
  | 'GitPullRequest'
  | 'Github'
  | 'Globe'
  | 'GraduationCap'
  | 'Heart'
  | 'Home'
  | 'HomeIcon'
  | 'Image'
  | 'KeyRound'
  | 'Landmark'
  | 'Laptop'
  | 'Layers'
  | 'Lock'
  | 'MemoryStick'
  | 'MessageSquare'
  | 'Mic'
  | 'Monitor'
  | 'Network'
  | 'RefreshCw'
  | 'Rocket'
  | 'Scale'
  | 'Search'
  | 'Server'
  | 'Settings'
  | 'Shield'
  | 'ShieldAlert'
  | 'ShieldCheck'
  | 'Shuffle'
  | 'Sliders'
  | 'Sparkles'
  | 'Star'
  | 'Target'
  | 'Terminal'
  | 'TestTube'
  | 'Timer'
  | 'TrendingDown'
  | 'TrendingUp'
  | 'Users'
  | 'Wifi'
  | 'Workflow'
  | 'Wrench'
  | 'X'
  | 'XCircle'
  | 'Zap'

const iconMap: Record<IconName, LucideIcon> = {
  Activity,
  AlertCircle,
  AlertTriangle,
  ArrowRight,
  Banknote,
  BarChart,
  BookOpen,
  Brain,
  Briefcase,
  Bug,
  Building,
  Calendar,
  Check,
  CheckCircle,
  Clock,
  Cloud,
  Code,
  Cpu,
  Database,
  DollarSign,
  Eye,
  Factory,
  FileCheck,
  FileCode,
  FileSearch,
  FileText,
  FileX,
  FlaskConical,
  Gamepad,
  Gauge,
  GitBranch,
  GitPullRequest,
  Github,
  Globe,
  GraduationCap,
  Heart,
  Home,
  HomeIcon,
  Image,
  KeyRound,
  Landmark,
  Laptop,
  Layers,
  Lock,
  MemoryStick,
  MessageSquare,
  Mic,
  Monitor,
  Network,
  RefreshCw,
  Rocket,
  Scale,
  Search,
  Server,
  Settings,
  Shield,
  ShieldAlert,
  ShieldCheck,
  Shuffle,
  Sliders,
  Sparkles,
  Star,
  Target,
  Terminal,
  TestTube,
  Timer,
  TrendingDown,
  TrendingUp,
  Users,
  Wifi,
  Workflow,
  Wrench,
  X,
  XCircle,
  Zap,
}

/**
 * Get a Lucide icon component by name
 * @param name - Icon name
 * @returns Lucide icon component
 */
export function getIcon(name: IconName): LucideIcon {
  return iconMap[name]
}

/**
 * Render an icon by name with optional className
 * @param name - Icon name
 * @param className - Optional CSS classes
 * @returns React element
 */
export function renderIcon(name: IconName, className?: string) {
  const Icon = getIcon(name)
  return <Icon className={className} />
}
