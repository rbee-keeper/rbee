// TEAM-421: Environment detection tests

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import {
  isTauriEnvironment,
  isNextJsEnvironment,
  isServerSide,
  isClientSide,
  getEnvironment,
  shouldUseTauriCommands,
  shouldUseDeepLinks,
  getActionStrategy,
} from './environment';

describe('Environment Detection', () => {
  // Store original values
  const originalWindow = global.window;
  const originalProcess = global.process;

  afterEach(() => {
    // Restore original values
    global.window = originalWindow;
    global.process = originalProcess;
  });

  describe('isTauriEnvironment', () => {
    it('returns true when __TAURI__ exists', () => {
      global.window = { __TAURI__: {} } as any;
      expect(isTauriEnvironment()).toBe(true);
    });

    it('returns false when __TAURI__ does not exist', () => {
      global.window = {} as any;
      expect(isTauriEnvironment()).toBe(false);
    });

    it('returns false in server-side context', () => {
      global.window = undefined as any;
      expect(isTauriEnvironment()).toBe(false);
    });
  });

  describe('isNextJsEnvironment', () => {
    it('returns true when window.next exists', () => {
      global.window = { next: {} } as any;
      expect(isNextJsEnvironment()).toBe(true);
    });

    it('returns true when NEXT_RUNTIME is set (SSR)', () => {
      global.window = undefined as any;
      global.process = { env: { NEXT_RUNTIME: 'nodejs' } } as any;
      expect(isNextJsEnvironment()).toBe(true);
    });

    it('returns false in Tauri environment', () => {
      global.window = { __TAURI__: {}, next: {} } as any;
      expect(isNextJsEnvironment()).toBe(false);
    });

    it('returns false in generic browser', () => {
      global.window = {} as any;
      expect(isNextJsEnvironment()).toBe(false);
    });
  });

  describe('getEnvironment', () => {
    it('returns "tauri" in Tauri environment', () => {
      global.window = { __TAURI__: {} } as any;
      expect(getEnvironment()).toBe('tauri');
    });

    it('returns "nextjs-ssg" in Next.js browser context', () => {
      global.window = { next: {} } as any;
      expect(getEnvironment()).toBe('nextjs-ssg');
    });

    it('returns "nextjs-ssr" in Next.js SSR context', () => {
      global.window = undefined as any;
      global.process = { env: { NEXT_RUNTIME: 'nodejs' } } as any;
      expect(getEnvironment()).toBe('nextjs-ssr');
    });

    it('returns "browser" in generic browser', () => {
      global.window = {} as any;
      expect(getEnvironment()).toBe('browser');
    });

    it('returns "server" in generic server context', () => {
      global.window = undefined as any;
      global.process = { env: {} } as any;
      expect(getEnvironment()).toBe('server');
    });
  });

  describe('Action Strategy', () => {
    it('uses tauri-command in Tauri environment', () => {
      global.window = { __TAURI__: {} } as any;
      expect(shouldUseTauriCommands()).toBe(true);
      expect(shouldUseDeepLinks()).toBe(false);
      expect(getActionStrategy()).toBe('tauri-command');
    });

    it('uses deep-link in Next.js SSG environment', () => {
      global.window = { next: {} } as any;
      expect(shouldUseTauriCommands()).toBe(false);
      expect(shouldUseDeepLinks()).toBe(true);
      expect(getActionStrategy()).toBe('deep-link');
    });

    it('uses deep-link in generic browser', () => {
      global.window = {} as any;
      expect(shouldUseTauriCommands()).toBe(false);
      expect(shouldUseDeepLinks()).toBe(true);
      expect(getActionStrategy()).toBe('deep-link');
    });

    it('uses none in server-side context', () => {
      global.window = undefined as any;
      expect(shouldUseTauriCommands()).toBe(false);
      expect(shouldUseDeepLinks()).toBe(false);
      expect(getActionStrategy()).toBe('none');
    });
  });
});
