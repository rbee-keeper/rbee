//! PKGBUILD parsing and execution
//!
//! TEAM-402: Migrated from rbee-hive
//!
//! This module handles:
//! - Parsing PKGBUILD files (Arch Linux package format)
//! - Executing build() and package() functions
//! - Fetching sources (git clone, tarballs, etc.)

mod executor;
mod parser;
mod source_fetcher;

pub use executor::{ExecutionError, PkgBuildExecutor};
pub use parser::{ParseError, PkgBuild};
pub use source_fetcher::fetch_sources;
