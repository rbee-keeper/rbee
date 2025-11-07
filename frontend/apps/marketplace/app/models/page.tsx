// TEAM-460: Models hub page - redirects to provider-specific pages
import { redirect } from 'next/navigation'

export default function ModelsPage() {
  // Redirect to HuggingFace models by default
  redirect('/models/huggingface')
}
