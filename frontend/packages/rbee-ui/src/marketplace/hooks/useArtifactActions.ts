// TEAM-421: Environment-aware action handlers for marketplace artifacts
// Automatically uses Tauri commands or deep links based on environment

import { getActionStrategy, getEnvironment } from '../../utils/environment';

/**
 * Get Tauri invoke function if available
 * Returns null if not in Tauri environment
 */
function getTauriInvoke(): ((cmd: string, args?: any) => Promise<any>) | null {
  if (typeof window === 'undefined') return null;
  if (!('__TAURI__' in window)) return null;
  
  // Access Tauri's invoke from window object
  // @ts-ignore - Tauri injects this at runtime
  return window.__TAURI__?.invoke || null;
}

export interface ArtifactActionHandlers {
  /**
   * Download a model
   * - Tauri: Direct download via ModelProvisioner
   * - Browser: Open rbee-keeper via deep link
   */
  downloadModel: (modelId: string) => Promise<void>;

  /**
   * Install a worker
   * - Tauri: Direct installation via Tauri command
   * - Browser: Open rbee-keeper via deep link
   */
  installWorker: (workerId: string) => Promise<void>;

  /**
   * Open external URL (HuggingFace, GitHub, etc.)
   */
  openExternal: (url: string) => void;

  /**
   * Check if actions are available in current environment
   */
  canPerformActions: boolean;

  /**
   * Get button label based on environment
   */
  getButtonLabel: (action: 'download' | 'install') => string;
}

export interface UseArtifactActionsOptions {
  /**
   * Callback when action starts
   */
  onActionStart?: (action: string) => void;

  /**
   * Callback when action succeeds
   */
  onActionSuccess?: (action: string) => void;

  /**
   * Callback when action fails
   */
  onActionError?: (action: string, error: Error) => void;
}

/**
 * Hook for environment-aware artifact actions
 * 
 * Automatically detects environment and uses appropriate action method:
 * - Tauri: Direct Rust commands via invoke()
 * - Next.js/Browser: Deep links (rbee://)
 * - SSR: Actions disabled
 * 
 * @example
 * ```tsx
 * function ModelDetailPage({ model }) {
 *   const actions = useArtifactActions({
 *     onActionSuccess: (action) => toast.success(`${action} started!`),
 *     onActionError: (action, error) => toast.error(error.message),
 *   });
 * 
 *   return (
 *     <button onClick={() => actions.downloadModel(model.id)}>
 *       {actions.getButtonLabel('download')}
 *     </button>
 *   );
 * }
 * ```
 */
export function useArtifactActions(
  options: UseArtifactActionsOptions = {}
): ArtifactActionHandlers {
  const { onActionStart, onActionSuccess, onActionError } = options;

  const strategy = getActionStrategy();
  const canPerformActions = strategy !== 'none';

  /**
   * Download model via appropriate method
   */
  const downloadModel = async (modelId: string): Promise<void> => {
    if (!canPerformActions) {
      throw new Error('Cannot perform actions in SSR/SSG environment');
    }

    onActionStart?.('download-model');

    try {
      switch (strategy) {
        case 'tauri-command':
          // Direct Tauri command
          const invoke = getTauriInvoke();
          if (!invoke) {
            throw new Error('Tauri invoke not available');
          }
          await invoke('model_download', {
            hiveId: 'localhost',
            modelId,
          });
          onActionSuccess?.('download-model');
          break;

        case 'deep-link':
          // Open rbee-keeper via deep link
          const downloadUrl = `rbee://download-model/${encodeURIComponent(modelId)}`;
          window.location.href = downloadUrl;
          onActionSuccess?.('download-model');
          break;

        default:
          throw new Error(`Unsupported action strategy: ${strategy}`);
      }
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      onActionError?.('download-model', err);
      throw err;
    }
  };

  /**
   * Install worker via appropriate method
   */
  const installWorker = async (workerId: string): Promise<void> => {
    if (!canPerformActions) {
      throw new Error('Cannot perform actions in SSR/SSG environment');
    }

    onActionStart?.('install-worker');

    try{
      switch (strategy) {
        case 'tauri-command':
          // Direct Tauri command
          const invoke = getTauriInvoke();
          if (!invoke) {
            throw new Error('Tauri invoke not available');
          }
          await invoke('marketplace_install_worker', {
            workerId,
          });
          onActionSuccess?.('install-worker');
          break;

        case 'deep-link':
          // Open rbee-keeper via deep link
          const installUrl = `rbee://install-worker/${encodeURIComponent(workerId)}`;
          window.location.href = installUrl;
          onActionSuccess?.('install-worker');
          break;

        default:
          throw new Error(`Unsupported action strategy: ${strategy}`);
      }
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      onActionError?.('install-worker', err);
      throw err;
    }
  };

  /**
   * Open external URL
   */
  const openExternal = (url: string): void => {
    if (typeof window === 'undefined') return;
    window.open(url, '_blank', 'noopener,noreferrer');
  };

  /**
   * Get button label based on environment
   */
  const getButtonLabel = (action: 'download' | 'install'): string => {
    if (strategy === 'tauri-command') {
      return action === 'download' ? 'Download Model' : 'Install Worker';
    }
    if (strategy === 'deep-link') {
      return 'Open in rbee App';
    }
    return action === 'download' ? 'Download' : 'Install';
  };

  return {
    downloadModel,
    installWorker,
    openExternal,
    canPerformActions,
    getButtonLabel,
  };
}

/**
 * Get environment-specific help text
 */
export function getEnvironmentHelpText(): string {
  const env = getEnvironment();

  switch (env) {
    case 'tauri':
      return 'Actions will download/install directly to your system.';
    case 'nextjs-ssg':
    case 'browser':
      return 'Actions will open the rbee app. Install rbee if you haven\'t already.';
    case 'nextjs-ssr':
    case 'server':
      return 'Actions are not available during server-side rendering.';
    default:
      return '';
  }
}
