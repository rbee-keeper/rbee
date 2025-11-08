// PKGBUILD validation tests
// Created by: TEAM-451

import { describe, it, expect } from 'vitest';
import { readFileSync, readdirSync } from 'fs';
import { join } from 'path';

const PKGBUILD_DIR = join(__dirname, '../public/pkgbuilds');

describe('PKGBUILD Files', () => {
  const pkgbuilds = readdirSync(PKGBUILD_DIR).filter(f => f.endsWith('.PKGBUILD'));

  it('should have PKGBUILD for all worker variants', () => {
    const requiredPKGBUILDs = [
      'llm-worker-rbee-cpu.PKGBUILD',
      'llm-worker-rbee-cuda.PKGBUILD',
      'llm-worker-rbee-metal.PKGBUILD',
      'sd-worker-rbee-cpu.PKGBUILD',
      'sd-worker-rbee-cuda.PKGBUILD',
    ];

    for (const required of requiredPKGBUILDs) {
      expect(pkgbuilds).toContain(required);
    }
  });

  describe.each(pkgbuilds)('%s', (pkgbuildFile) => {
    const content = readFileSync(join(PKGBUILD_DIR, pkgbuildFile), 'utf-8');

    it('should have required metadata fields', () => {
      expect(content).toMatch(/^pkgname=/m);
      expect(content).toMatch(/^pkgver=/m);
      expect(content).toMatch(/^pkgrel=/m);
      expect(content).toMatch(/^pkgdesc=/m);
      expect(content).toMatch(/^arch=/m);
      expect(content).toMatch(/^url=/m);
      expect(content).toMatch(/^license=/m);
    });

    it('should have build() function', () => {
      expect(content).toMatch(/^build\(\)/m);
      expect(content).toContain('cargo build');
    });

    it('should have package() function', () => {
      expect(content).toMatch(/^package\(\)/m);
      expect(content).toContain('install');
    });

    it('should have check() function for tests', () => {
      expect(content).toMatch(/^check\(\)/m);
      expect(content).toContain('cargo test');
    });

    it('should support GitHub releases (not just git)', () => {
      // Should have source array with both git and release options
      expect(content).toMatch(/source=/);
      
      // Should have logic to use GitHub releases when available
      // Either via commented alternative or conditional logic
      const hasReleaseSupport = 
        content.includes('github.com') && 
        (content.includes('releases/download') || content.includes('# Release'));
      
      expect(hasReleaseSupport).toBe(true);
    });

    it('should have correct feature flags', () => {
      const pkgname = content.match(/pkgname=([^\n]+)/)?.[1];
      
      if (pkgname?.includes('cpu')) {
        expect(content).toMatch(/--features cpu/);
      } else if (pkgname?.includes('cuda')) {
        expect(content).toMatch(/--features cuda/);
      } else if (pkgname?.includes('metal')) {
        expect(content).toMatch(/--features metal/);
      }
    });

    it('should have correct dependencies', () => {
      const pkgname = content.match(/pkgname=([^\n]+)/)?.[1];
      
      // All should have gcc
      expect(content).toMatch(/depends=.*gcc/);
      
      // CUDA variants should have cuda dependency
      if (pkgname?.includes('cuda')) {
        expect(content).toMatch(/depends=.*cuda/);
      }
    });

    it('should install to correct location', () => {
      const pkgname = content.match(/pkgname=([^\n]+)/)?.[1];
      
      // Should install with the package name
      expect(content).toContain(`$pkgdir/usr/local/bin/$pkgname`);
    });

    it('should have correct architecture support', () => {
      const pkgname = content.match(/pkgname=([^\n]+)/)?.[1];
      const archLine = content.match(/arch=\(([^)]+)\)/)?.[1];
      
      // CPU variants should support both x86_64 and aarch64
      if (pkgname?.includes('cpu')) {
        expect(archLine).toContain('x86_64');
        expect(archLine).toContain('aarch64');
      }
      
      // CUDA variants should only support x86_64
      if (pkgname?.includes('cuda')) {
        expect(archLine).toContain('x86_64');
        expect(archLine).not.toContain('aarch64');
      }
      
      // Metal variants should only support aarch64
      if (pkgname?.includes('metal')) {
        expect(archLine).toContain('aarch64');
        expect(archLine).not.toContain('x86_64');
      }
    });

    it('should use workspace-level target directory', () => {
      // Should reference target/release at workspace root
      expect(content).toContain('target/release');
      
      // Should NOT use bin-specific target
      expect(content).not.toContain('bin/30_llm_worker_rbee/target');
      expect(content).not.toContain('bin/31_sd_worker_rbee/target');
    });

    it('should have GPL-3.0-or-later license', () => {
      expect(content).toMatch(/license=.*GPL-3\.0-or-later/);
    });

    it('should point to correct repository', () => {
      expect(content).toContain('github.com/veighnsche/llama-orch');
    });
  });

  describe('Worker-specific PKGBUILDs', () => {
    it('LLM workers should build from bin/30_llm_worker_rbee', () => {
      const llmPKGBUILDs = pkgbuilds.filter(f => f.startsWith('llm-worker'));
      
      for (const pkgbuild of llmPKGBUILDs) {
        const content = readFileSync(join(PKGBUILD_DIR, pkgbuild), 'utf-8');
        expect(content).toContain('bin/30_llm_worker_rbee');
      }
    });

    it('SD workers should build from bin/31_sd_worker_rbee', () => {
      const sdPKGBUILDs = pkgbuilds.filter(f => f.startsWith('sd-worker'));
      
      for (const pkgbuild of sdPKGBUILDs) {
        const content = readFileSync(join(PKGBUILD_DIR, pkgbuild), 'utf-8');
        expect(content).toContain('bin/31_sd_worker_rbee');
      }
    });
  });
});
