// PKGBUILD validation tests
// Created by: TEAM-451

import { readdirSync, readFileSync } from 'fs'
import { join } from 'path'
import { describe, expect, it } from 'vitest'

const PKGBUILD_DIR = join(__dirname, '../public/pkgbuilds/arch')

describe('PKGBUILD Files', () => {
  const pkgbuilds = readdirSync(PKGBUILD_DIR).filter((f) => f.endsWith('.PKGBUILD'))

  it('should have PKGBUILD for all workers (bin + git)', () => {
    const requiredPKGBUILDs = [
      'llm-worker-rbee-bin.PKGBUILD',
      'llm-worker-rbee-git.PKGBUILD',
      'sd-worker-rbee-bin.PKGBUILD',
      'sd-worker-rbee-git.PKGBUILD',
    ]

    for (const required of requiredPKGBUILDs) {
      expect(pkgbuilds).toContain(required)
    }
  })

  describe.each(pkgbuilds)('%s', (pkgbuildFile) => {
    const content = readFileSync(join(PKGBUILD_DIR, pkgbuildFile), 'utf-8')

    it('should have required metadata fields', () => {
      expect(content).toMatch(/^pkgname=/m)
      expect(content).toMatch(/^pkgver=/m)
      expect(content).toMatch(/^pkgrel=/m)
      expect(content).toMatch(/^pkgdesc=/m)
      expect(content).toMatch(/^arch=/m)
      expect(content).toMatch(/^url=/m)
      expect(content).toMatch(/^license=/m)
    })

    it('should have build() function for git versions', () => {
      const pkgname = content.match(/pkgname=([^\n]+)/)?.[1]
      
      if (pkgname?.includes('-git')) {
        expect(content).toMatch(/^build\(\)/m)
        expect(content).toContain('cargo build')
      }
    })

    it('should have package() function', () => {
      expect(content).toMatch(/^package\(\)/m)
      expect(content).toContain('install')
    })

    it('should have check() function for git versions', () => {
      const pkgname = content.match(/pkgname=([^\n]+)/)?.[1]
      
      if (pkgname?.includes('-git')) {
        expect(content).toMatch(/^check\(\)/m)
        expect(content).toContain('cargo test')
      }
    })

    it('should support GitHub releases or git', () => {
      // Should support downloading from releases OR building from git
      const hasSupport =
        content.includes('github.com') && (content.includes('releases') || content.includes('git+'))

      expect(hasSupport).toBe(true)
    })

    it('should have correct feature flags', () => {
      const pkgname = content.match(/pkgname=([^\n]+)/)?.[1]

      if (pkgname?.includes('cpu')) {
        expect(content).toMatch(/--features cpu/)
      } else if (pkgname?.includes('cuda')) {
        expect(content).toMatch(/--features cuda/)
      } else if (pkgname?.includes('metal')) {
        expect(content).toMatch(/--features metal/)
      }
    })

    it('should have correct dependencies', () => {
      const pkgname = content.match(/pkgname=([^\n]+)/)?.[1]

      // All should have gcc
      expect(content).toMatch(/depends=.*gcc/)

      // CUDA variants should have cuda dependency
      if (pkgname?.includes('cuda')) {
        expect(content).toMatch(/depends=.*cuda/)
      }
    })

    it('should install to correct location', () => {
      // Should install to /usr/local/bin
      expect(content).toContain('$pkgdir/usr/local/bin/')
    })

    it('should have correct architecture support', () => {
      const archLine = content.match(/arch=\(([^)]+)\)/)?.[1]

      // All bin/git versions support both architectures
      expect(archLine).toContain('x86_64')
      expect(archLine).toContain('aarch64')
    })

    it('should use workspace-level target directory for git versions', () => {
      const pkgname = content.match(/pkgname=([^\n]+)/)?.[1]
      
      // Git versions should reference target/release at workspace root
      if (pkgname?.includes('-git')) {
        expect(content).toContain('target/release')
        
        // Should NOT use bin-specific target
        expect(content).not.toContain('bin/30_llm_worker_rbee/target')
        expect(content).not.toContain('bin/31_sd_worker_rbee/target')
      }
    })

    it('should have GPL-3.0-or-later license', () => {
      expect(content).toMatch(/license=.*GPL-3\.0-or-later/)
    })

    it('should point to correct repository', () => {
      expect(content).toContain('github.com/rbee-keeper/rbee')
    })
  })

  describe('Worker-specific PKGBUILDs', () => {
    it('LLM git workers should build from bin/30_llm_worker_rbee', () => {
      const llmGitPKGBUILDs = pkgbuilds.filter((f) => f.startsWith('llm-worker') && f.includes('-git'))

      for (const pkgbuild of llmGitPKGBUILDs) {
        const content = readFileSync(join(PKGBUILD_DIR, pkgbuild), 'utf-8')
        expect(content).toContain('bin/30_llm_worker_rbee')
      }
    })

    it('SD git workers should build from bin/31_sd_worker_rbee', () => {
      const sdGitPKGBUILDs = pkgbuilds.filter((f) => f.startsWith('sd-worker') && f.includes('-git'))

      for (const pkgbuild of sdGitPKGBUILDs) {
        const content = readFileSync(join(PKGBUILD_DIR, pkgbuild), 'utf-8')
        expect(content).toContain('bin/31_sd_worker_rbee')
      }
    })
    
    it('Binary versions should auto-detect platform', () => {
      const binPKGBUILDs = pkgbuilds.filter((f) => f.includes('-bin'))

      for (const pkgbuild of binPKGBUILDs) {
        const content = readFileSync(join(PKGBUILD_DIR, pkgbuild), 'utf-8')
        expect(content).toContain('_detect_platform')
      }
    })
  })
})
