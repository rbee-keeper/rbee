// TEAM-390: HTTP server for SD worker (placeholder)
// TEAM-394: Complete HTTP infrastructure implementation
// TEAM-395: DELETED - Wrong approach (bypassed operations-contract)
// TEAM-396: RESTORED - Now using operations-contract correctly
//
// Provides HTTP infrastructure with operations-contract integration.

pub mod backend;
pub mod health;
pub mod jobs;    // TEAM-396: Job submission (operations-contract)
pub mod ready;
pub mod routes;
pub mod server;
pub mod stream;  // TEAM-396: SSE streaming

// Re-export commonly used types
pub use backend::AppState;
pub use routes::create_router;
pub use server::{HttpServer, ServerError};
