//! Worker Provisioner - Downloads/builds workers from PKGBUILDs
//!
//! TEAM-402: Migrated from rbee-hive to dedicated crate
//!
//! This crate handles worker installation from PKGBUILD files, supporting:
//! - Source builds (git clone + cargo build)
//! - Binary packages (pre-built tarballs)
//! - Architecture-specific sources (source_x86_64, source_aarch64)
//! - Cancellable operations
//!
//! # Architecture
//!
//! ```text
//! WorkerProvisioner
//!     ↓
//! CatalogClient → Fetch PKGBUILD
//!     ↓
//! PKGBUILD Parser → Parse metadata
//!     ↓
//! Source Fetcher → Download sources
//!     ↓
//! PKGBUILD Executor → Build & Package
//!     ↓
//! Worker installation complete
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use rbee_hive_worker_provisioner::WorkerProvisioner;
//! use rbee_hive_artifact_catalog::ArtifactProvisioner;
//! use tokio_util::sync::CancellationToken;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let provisioner = WorkerProvisioner::new()?;
//! let cancel_token = CancellationToken::new();
//! let worker = provisioner.provision(
//!     "llm-worker-rbee-cpu",
//!     "job-123",
//!     cancel_token
//! ).await?;
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

mod catalog_client;
mod pkgbuild;
mod provisioner;

// TEAM-402: AUR binary support (Phase 4)
// mod aur;

pub use catalog_client::CatalogClient;
pub use pkgbuild::{PkgBuild, PkgBuildExecutor, ParseError, ExecutionError};
pub use provisioner::WorkerProvisioner;
