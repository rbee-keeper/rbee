// TEAM-460: Legacy redirect - models now organized by provider
import { redirect } from 'next/navigation'

interface Props {
  params: Promise<{ slug: string }>
}

export default async function ModelPage({ params }: Props) {
  const { slug } = await params
  
  // Check if it's a Civitai model (starts with civitai-)
  if (slug.startsWith('civitai-')) {
    redirect(`/models/civitai/${slug}`)
  }
  
  // Default to HuggingFace
  redirect(`/models/huggingface/${slug}`)
}
