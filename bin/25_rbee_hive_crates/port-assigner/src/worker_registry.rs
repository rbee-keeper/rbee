// TEAM-XXX: Worker registry for tracking port assignments
//
// Maps worker PIDs to their assigned ports for cleanup

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Registry for tracking worker port assignments
///
/// Maintains a bidirectional mapping between PIDs and ports
/// to enable port cleanup when workers are terminated.
///
/// # Thread Safety
/// This component is thread-safe and can be shared across async tasks.
///
/// # Example
/// ```
/// use port_assigner::WorkerRegistry;
///
/// let registry = WorkerRegistry::new();
///
/// // Register worker
/// registry.register(12345, 8080);
///
/// // Lookup port by PID
/// assert_eq!(registry.get_port(12345), Some(8080));
///
/// // Unregister when worker stops
/// registry.unregister(12345);
/// ```
#[derive(Debug, Clone)]
pub struct WorkerRegistry {
    state: Arc<Mutex<RegistryState>>,
}

#[derive(Debug)]
struct RegistryState {
    /// PID -> Port mapping
    pid_to_port: HashMap<u32, u16>,
    /// Port -> PID mapping (for reverse lookup)
    port_to_pid: HashMap<u16, u32>,
}

impl Default for WorkerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkerRegistry {
    /// Create a new worker registry
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(RegistryState {
                pid_to_port: HashMap::new(),
                port_to_pid: HashMap::new(),
            })),
        }
    }

    /// Register a worker with its assigned port
    ///
    /// # Arguments
    /// * `pid` - Process ID of the worker
    /// * `port` - Assigned port number
    ///
    /// # Example
    /// ```
    /// use port_assigner::WorkerRegistry;
    ///
    /// let registry = WorkerRegistry::new();
    /// registry.register(12345, 8080);
    /// ```
    pub fn register(&self, pid: u32, port: u16) {
        let mut state = self.state.lock().unwrap();
        state.pid_to_port.insert(pid, port);
        state.port_to_pid.insert(port, pid);
    }

    /// Unregister a worker by PID
    ///
    /// Returns the port that was assigned to this worker, if any.
    ///
    /// # Arguments
    /// * `pid` - Process ID of the worker
    ///
    /// # Returns
    /// The port number if the worker was registered, `None` otherwise
    ///
    /// # Example
    /// ```
    /// use port_assigner::WorkerRegistry;
    ///
    /// let registry = WorkerRegistry::new();
    /// registry.register(12345, 8080);
    ///
    /// let port = registry.unregister(12345);
    /// assert_eq!(port, Some(8080));
    /// ```
    pub fn unregister(&self, pid: u32) -> Option<u16> {
        let mut state = self.state.lock().unwrap();
        if let Some(port) = state.pid_to_port.remove(&pid) {
            state.port_to_pid.remove(&port);
            Some(port)
        } else {
            None
        }
    }

    /// Get the port assigned to a worker
    ///
    /// # Arguments
    /// * `pid` - Process ID of the worker
    ///
    /// # Returns
    /// The port number if the worker is registered, `None` otherwise
    ///
    /// # Example
    /// ```
    /// use port_assigner::WorkerRegistry;
    ///
    /// let registry = WorkerRegistry::new();
    /// registry.register(12345, 8080);
    ///
    /// assert_eq!(registry.get_port(12345), Some(8080));
    /// assert_eq!(registry.get_port(99999), None);
    /// ```
    #[must_use]
    pub fn get_port(&self, pid: u32) -> Option<u16> {
        let state = self.state.lock().unwrap();
        state.pid_to_port.get(&pid).copied()
    }

    /// Get the PID of the worker using a specific port
    ///
    /// # Arguments
    /// * `port` - Port number
    ///
    /// # Returns
    /// The PID if a worker is using this port, `None` otherwise
    ///
    /// # Example
    /// ```
    /// use port_assigner::WorkerRegistry;
    ///
    /// let registry = WorkerRegistry::new();
    /// registry.register(12345, 8080);
    ///
    /// assert_eq!(registry.get_pid(8080), Some(12345));
    /// assert_eq!(registry.get_pid(9999), None);
    /// ```
    #[must_use]
    pub fn get_pid(&self, port: u16) -> Option<u32> {
        let state = self.state.lock().unwrap();
        state.port_to_pid.get(&port).copied()
    }

    /// Get the count of registered workers
    ///
    /// # Returns
    /// Number of workers currently registered
    ///
    /// # Example
    /// ```
    /// use port_assigner::WorkerRegistry;
    ///
    /// let registry = WorkerRegistry::new();
    /// assert_eq!(registry.count(), 0);
    ///
    /// registry.register(12345, 8080);
    /// assert_eq!(registry.count(), 1);
    /// ```
    #[must_use]
    pub fn count(&self) -> usize {
        let state = self.state.lock().unwrap();
        state.pid_to_port.len()
    }

    /// Get all registered workers
    ///
    /// Returns a vector of (PID, port) tuples.
    ///
    /// # Returns
    /// Vector of (PID, port) pairs
    ///
    /// # Example
    /// ```
    /// use port_assigner::WorkerRegistry;
    ///
    /// let registry = WorkerRegistry::new();
    /// registry.register(12345, 8080);
    /// registry.register(12346, 8081);
    ///
    /// let workers = registry.list();
    /// assert_eq!(workers.len(), 2);
    /// ```
    #[must_use]
    pub fn list(&self) -> Vec<(u32, u16)> {
        let state = self.state.lock().unwrap();
        state.pid_to_port.iter().map(|(&pid, &port)| (pid, port)).collect()
    }

    /// Clear all registrations
    ///
    /// Removes all worker registrations from the registry.
    ///
    /// # Example
    /// ```
    /// use port_assigner::WorkerRegistry;
    ///
    /// let registry = WorkerRegistry::new();
    /// registry.register(12345, 8080);
    /// registry.register(12346, 8081);
    ///
    /// registry.clear();
    /// assert_eq!(registry.count(), 0);
    /// ```
    pub fn clear(&self) {
        let mut state = self.state.lock().unwrap();
        state.pid_to_port.clear();
        state.port_to_pid.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_get() {
        let registry = WorkerRegistry::new();

        registry.register(12345, 8080);
        assert_eq!(registry.get_port(12345), Some(8080));
        assert_eq!(registry.get_pid(8080), Some(12345));
    }

    #[test]
    fn test_unregister() {
        let registry = WorkerRegistry::new();

        registry.register(12345, 8080);
        let port = registry.unregister(12345);

        assert_eq!(port, Some(8080));
        assert_eq!(registry.get_port(12345), None);
        assert_eq!(registry.get_pid(8080), None);
    }

    #[test]
    fn test_unregister_nonexistent() {
        let registry = WorkerRegistry::new();

        let port = registry.unregister(99999);
        assert_eq!(port, None);
    }

    #[test]
    fn test_count() {
        let registry = WorkerRegistry::new();

        assert_eq!(registry.count(), 0);

        registry.register(12345, 8080);
        assert_eq!(registry.count(), 1);

        registry.register(12346, 8081);
        assert_eq!(registry.count(), 2);

        registry.unregister(12345);
        assert_eq!(registry.count(), 1);
    }

    #[test]
    fn test_list() {
        let registry = WorkerRegistry::new();

        registry.register(12345, 8080);
        registry.register(12346, 8081);

        let workers = registry.list();
        assert_eq!(workers.len(), 2);
        assert!(workers.contains(&(12345, 8080)));
        assert!(workers.contains(&(12346, 8081)));
    }

    #[test]
    fn test_clear() {
        let registry = WorkerRegistry::new();

        registry.register(12345, 8080);
        registry.register(12346, 8081);

        registry.clear();
        assert_eq!(registry.count(), 0);
        assert_eq!(registry.get_port(12345), None);
        assert_eq!(registry.get_pid(8080), None);
    }

    #[test]
    fn test_thread_safety() {
        use std::thread;

        let registry = WorkerRegistry::new();
        let mut handles = vec![];

        // Spawn 10 threads, each registering 10 workers
        for i in 0..10 {
            let registry_clone = registry.clone();
            let handle = thread::spawn(move || {
                for j in 0..10 {
                    let pid = (i * 10 + j) as u32;
                    let port = 8080 + pid as u16;
                    registry_clone.register(pid, port);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Should have 100 registered workers
        assert_eq!(registry.count(), 100);
    }
}
