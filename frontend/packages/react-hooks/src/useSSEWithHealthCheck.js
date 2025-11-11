/**
 * TEAM-356: SSE connection with health check
 *
 * Prevents CORS errors by checking health before connecting to SSE stream.
 * Includes automatic retry logic and cleanup.
 */
import { useEffect, useRef, useState } from 'react';
/**
 * Hook for SSE connection with health check
 *
 * Automatically handles:
 * - Health check before SSE connection
 * - Connection state management
 * - Auto-retry on failure
 * - Cleanup on unmount
 *
 * @param createMonitor - Factory function to create monitor
 * @param baseUrl - Base URL for the service
 * @param options - Optional configuration
 * @returns Object with data, connected, loading, error, and retry
 *
 * @example
 * ```typescript
 * const { data, connected, error } = useSSEWithHealthCheck(
 *   (baseUrl) => new sdk.HeartbeatMonitor(baseUrl),
 *   'http://localhost:7833'
 * )
 * ```
 */
export function useSSEWithHealthCheck(createMonitor, baseUrl, options = {}) {
    const { autoRetry = true, retryDelayMs = 5000, maxRetries = 3 } = options;
    const [data, setData] = useState(null);
    const [connected, setConnected] = useState(false);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const mountedRef = useRef(true);
    const monitorRef = useRef(null);
    const retriesRef = useRef(0);
    const startMonitoring = async () => {
        if (!mountedRef.current)
            return;
        setLoading(true);
        setError(null);
        try {
            const monitor = createMonitor(baseUrl);
            monitorRef.current = monitor;
            // Check health before starting SSE
            const isHealthy = await monitor.checkHealth();
            if (!mountedRef.current)
                return;
            if (!isHealthy) {
                throw new Error('Service is offline');
            }
            // Start SSE connection
            monitor.start((snapshot) => {
                if (!mountedRef.current)
                    return;
                setData(snapshot);
                setConnected(true);
                setError(null);
                setLoading(false);
                retriesRef.current = 0; // Reset retry count on success
            });
        }
        catch (err) {
            const error = err;
            if (!mountedRef.current)
                return;
            setError(error);
            setConnected(false);
            setLoading(false);
            // Auto-retry if enabled
            if (autoRetry && retriesRef.current < maxRetries) {
                retriesRef.current++;
                console.warn(`[useSSEWithHealthCheck] Retry ${retriesRef.current}/${maxRetries} in ${retryDelayMs}ms`);
                setTimeout(startMonitoring, retryDelayMs);
            }
        }
    };
    useEffect(() => {
        mountedRef.current = true;
        retriesRef.current = 0;
        startMonitoring();
        return () => {
            mountedRef.current = false;
            monitorRef.current?.stop();
        };
    }, [baseUrl]);
    const retry = () => {
        retriesRef.current = 0;
        startMonitoring();
    };
    return { data, connected, loading, error, retry };
}
//# sourceMappingURL=useSSEWithHealthCheck.js.map