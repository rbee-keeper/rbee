// TEAM-453: Unit tests for InstallButton component
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { InstallButton } from '@/components/InstallButton'

describe('InstallButton', () => {
  it('should render button element', () => {
    render(<InstallButton modelId="llama-3.1-8b" />)
    const button = screen.getByRole('button')
    expect(button).toBeInTheDocument()
  })

  it('should show loading state initially', () => {
    render(<InstallButton modelId="llama-3.1-70b" />)
    // Button shows "Checking..." while loading
    expect(screen.getByText(/checking/i)).toBeInTheDocument()
  })

  it('should be disabled while loading', () => {
    render(<InstallButton modelId="sdxl-turbo" />)
    const button = screen.getByRole('button')
    expect(button).toBeDisabled()
  })

  it('should have loading spinner', () => {
    render(<InstallButton modelId="mistral-7b" />)
    const button = screen.getByRole('button')
    // Check for spinner SVG
    const svg = button.querySelector('svg')
    expect(svg).toBeInTheDocument()
  })
})
