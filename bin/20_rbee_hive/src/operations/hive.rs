//! Hive management operations
//!
//! TEAM-388: Extracted from job_router.rs
//!
//! Handles:
//! - HiveCheck: Narration test through hive SSE

use anyhow::Result;
use observability_narration_core::n;
use operations_contract::Operation;

/// Handle hive-related operations
///
/// TEAM-388: Extracted from job_router.rs for better organization
pub async fn handle_hive_operation(operation: &Operation) -> Result<()> {
    match operation {
        // TEAM-313: HiveCheck - narration test through hive SSE
        // TEAM-314: Migrated to n!() macro
        // TEAM-381: Narration context now set at router level, no need to set here
        Operation::HiveCheck { .. } => {
            n!("hive_check_start", "🔍 Starting hive narration check");
            crate::hive_check::handle_hive_check().await?;
            n!("hive_check_complete", "✅ Hive narration check complete");
            Ok(())
        }
        _ => Err(anyhow::anyhow!("Not a hive operation")),
    }
}
