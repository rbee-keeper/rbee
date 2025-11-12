// Created by: TEAM-394
// TEAM-394: HTTP server lifecycle management with graceful shutdown

use axum::Router;
use std::net::SocketAddr;
use thiserror::Error;
use tokio::net::TcpListener;
use tracing::{error, info, warn};

/// HTTP server errors
#[derive(Debug, Error)]
pub enum ServerError {
    /// Failed to bind to the specified address
    #[error("Failed to bind to {addr}: {source}")]
    BindFailed { addr: SocketAddr, source: std::io::Error },

    /// Server runtime error
    #[error("Server runtime error: {0}")]
    Runtime(String),
}

/// HTTP server with lifecycle management
///
/// Manages the complete lifecycle of the Axum HTTP server:
/// - Binding to a configurable address
/// - Graceful shutdown on SIGTERM/SIGINT
/// - Error propagation for bind failures
///
/// # Graceful Shutdown
/// The server handles both SIGTERM (Kubernetes/Docker) and SIGINT (Ctrl+C).
/// In-flight requests are allowed to complete before shutdown.
pub struct HttpServer {
    /// Bind address
    addr: SocketAddr,

    /// Router with all endpoints
    router: Router,
}

impl HttpServer {
    /// Create new HTTP server
    ///
    /// # Arguments
    /// * `addr` - Socket address to bind to (e.g., `0.0.0.0:8080`)
    /// * `router` - Axum router with all endpoints configured
    ///
    /// # Returns
    /// * `Ok(HttpServer)` - Server ready to run
    ///
    /// # Example
    /// ```no_run
    /// use std::net::SocketAddr;
    /// use axum::Router;
    /// # use sd_worker_rbee::http::server::HttpServer;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let addr: SocketAddr = "0.0.0.0:8080".parse()?;
    /// let router = Router::new();
    /// let server = HttpServer::new(addr, router);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(addr: SocketAddr, router: Router) -> Self {
        info!(addr = %addr, "HTTP server initialized");
        Self { addr, router }
    }

    /// Run server until shutdown signal received
    ///
    /// This method blocks until:
    /// - SIGTERM is received (Kubernetes/Docker)
    /// - SIGINT is received (Ctrl+C)
    ///
    /// The server will complete in-flight requests before shutting down.
    ///
    /// # Returns
    /// * `Ok(())` - Server shut down gracefully
    /// * `Err(ServerError)` - Server encountered an error
    ///
    /// # Example
    /// ```no_run
    /// # use sd_worker_rbee::http::server::HttpServer;
    /// # use std::net::SocketAddr;
    /// # use axum::Router;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let addr: SocketAddr = "0.0.0.0:8080".parse()?;
    /// let router = Router::new();
    /// let server = HttpServer::new(addr, router);
    /// server.run().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn run(self) -> Result<(), ServerError> {
        let listener = TcpListener::bind(self.addr).await.map_err(|source| {
            error!(addr = %self.addr, error = %source, "Failed to bind");
            ServerError::BindFailed { addr: self.addr, source }
        })?;

        info!(addr = %self.addr, "HTTP server listening");

        // Run server with graceful shutdown
        axum::serve(listener, self.router)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|e| ServerError::Runtime(e.to_string()))?;

        info!("HTTP server shutdown complete");

        Ok(())
    }

    /// Get the bind address
    #[must_use]
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }
}

/// Wait for shutdown signal (SIGTERM or SIGINT)
///
/// This function waits for either:
/// - SIGTERM (sent by Kubernetes/Docker during graceful shutdown)
/// - SIGINT (sent by Ctrl+C during development)
///
/// # Platform Support
/// - Unix: Handles both SIGTERM and SIGINT
/// - Windows: Only handles Ctrl+C (SIGINT)
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c().await.expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {
            warn!("Received SIGINT (Ctrl+C), initiating graceful shutdown");
        },
        () = terminate => {
            warn!("Received SIGTERM, initiating graceful shutdown");
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{routing::get, Json};
    use serde::Serialize;

    #[derive(Serialize)]
    struct TestResponse {
        status: String,
    }

    async fn test_handler() -> Json<TestResponse> {
        Json(TestResponse { status: "ok".to_string() })
    }

    #[tokio::test]
    async fn test_server_creation() {
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let router = Router::new().route("/test", get(test_handler));

        let server = HttpServer::new(addr, router);
        assert_eq!(server.addr(), addr);
    }

    #[tokio::test]
    async fn test_server_error_display() {
        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        let io_error = std::io::Error::new(std::io::ErrorKind::AddrInUse, "port in use");

        let error = ServerError::BindFailed { addr, source: io_error };

        let error_msg = error.to_string();
        assert!(error_msg.contains("127.0.0.1:8080"));
        assert!(error_msg.contains("Failed to bind"));
    }

    #[tokio::test]
    async fn test_runtime_error() {
        let error = ServerError::Runtime("test error".to_string());
        assert_eq!(error.to_string(), "Server runtime error: test error");
    }
}
