// TEAM-476: Redirect /models to /models/civitai for MVP
import { redirect } from 'next/navigation'

export default function ModelsPage() {
  redirect('/models/civitai')
}
