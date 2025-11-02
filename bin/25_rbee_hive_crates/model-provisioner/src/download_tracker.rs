//! Download progress tracking and heartbeat
//!
//! Reusable across all vendor sources (HuggingFace, GitHub, etc.)

use observability_narration_core::n;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

/// Download progress tracker
///
/// Provides:
/// - Progress reporting (bytes downloaded)
/// - Heartbeat messages (periodic "still downloading...")
/// - Cancellation support
/// - Narration with proper job_id context
#[derive(Clone, Debug)]
pub struct DownloadTracker {
    /// Total bytes downloaded
    bytes_downloaded: Arc<AtomicU64>,
    /// Expected total size (if known)
    total_size: Option<u64>,
    /// TEAM-379: Percentage (stored as 0-10000 for 0.00%-100.00%)
    percentage: Arc<AtomicU64>,
    /// Cancellation token
    cancel_token: CancellationToken,
    /// Progress sender
    progress_tx: watch::Sender<DownloadProgress>,
    /// Job ID for narration routing
    job_id: String,
}

/// Download progress snapshot
#[derive(Debug, Clone, Copy)]
pub struct DownloadProgress {
    /// Bytes downloaded so far
    pub bytes_downloaded: u64,
    /// Total size if known
    pub total_size: Option<u64>,
    /// Percentage complete (0.0-100.0)
    pub percentage: Option<f64>,
}

impl DownloadTracker {
    /// Create a new download tracker
    ///
    /// # Arguments
    /// * `job_id` - Job ID for narration routing
    /// * `total_size` - Expected total size (if known)
    /// * `cancel_token` - Cancellation token from job registry
    pub fn new(
        job_id: String,
        total_size: Option<u64>,
        cancel_token: CancellationToken,
    ) -> (Self, watch::Receiver<DownloadProgress>) {
        let (progress_tx, progress_rx) = watch::channel(DownloadProgress {
            bytes_downloaded: 0,
            total_size,
            percentage: None,
        });

        let tracker = Self {
            bytes_downloaded: Arc::new(AtomicU64::new(0)),
            total_size,
            percentage: Arc::new(AtomicU64::new(0)),  // TEAM-379: Start at 0%
            cancel_token,
            progress_tx,
            job_id,
        };

        (tracker, progress_rx)
    }

    /// Get cancellation token
    pub fn cancel_token(&self) -> CancellationToken {
        self.cancel_token.clone()
    }

    /// Update bytes downloaded
    pub fn update_progress(&self, bytes: u64) {
        self.bytes_downloaded.store(bytes, Ordering::Relaxed);
        
        let percentage = self.total_size.map(|total| {
            if total > 0 {
                (bytes as f64 / total as f64) * 100.0
            } else {
                0.0
            }
        });

        let progress = DownloadProgress {
            bytes_downloaded: bytes,
            total_size: self.total_size,
            percentage,
        };

        let _ = self.progress_tx.send(progress);
    }

    /// Increment bytes downloaded
    pub fn increment_progress(&self, bytes: u64) {
        let new_total = self.bytes_downloaded.fetch_add(bytes, Ordering::Relaxed) + bytes;
        self.update_progress(new_total);
    }

    /// TEAM-379: Update percentage from progress callback
    ///
    /// # Arguments
    /// * `pct` - Percentage as 0.0-1.0 (e.g., 0.365 = 36.5%)
    pub fn update_percentage(&self, pct: f64) {
        // Store as 0-10000 for precision (36.5% = 3650)
        let pct_int = (pct * 10000.0) as u64;
        self.percentage.store(pct_int, Ordering::Relaxed);
    }

    /// Get current progress
    pub fn current_progress(&self) -> DownloadProgress {
        let bytes = self.bytes_downloaded.load(Ordering::Relaxed);
        
        // TEAM-379: Use percentage from callback if available
        let pct_int = self.percentage.load(Ordering::Relaxed);
        let percentage = if pct_int > 0 {
            Some(pct_int as f64 / 100.0)  // Convert 3650 -> 36.50%
        } else {
            // Fallback to calculating from bytes if we have total_size
            self.total_size.map(|total| {
                if total > 0 {
                    (bytes as f64 / total as f64) * 100.0
                } else {
                    0.0
                }
            })
        };

        DownloadProgress {
            bytes_downloaded: bytes,
            total_size: self.total_size,
            percentage,
        }
    }

    /// Start heartbeat task
    ///
    /// Sends periodic progress updates via narration.
    /// Automatically stops when download completes or is cancelled.
    pub fn start_heartbeat(&self, artifact_name: String) -> tokio::task::JoinHandle<()> {
        let tracker = self.clone();
        let cancel_token = self.cancel_token.clone();
        let job_id = self.job_id.clone();

        tokio::spawn(async move {
            // CRITICAL: Set up narration context for spawned task
            use observability_narration_core::context;
            let ctx = context::NarrationContext::new().with_job_id(&job_id);
            context::with_narration_context(ctx, async move {
                let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(10));
                interval.tick().await; // Skip first immediate tick

                loop {
                    tokio::select! {
                        _ = cancel_token.cancelled() => {
                            n!("download_cancelled", "❌ Download cancelled");
                            break;
                        }
                        _ = interval.tick() => {
                            let progress = tracker.current_progress();
                            
                            // TEAM-379: Show percentage if available from progress callback
                            if let Some(pct) = progress.percentage {
                                n!(
                                    "download_progress",
                                    "⏳ Still downloading {} ({:.1}% complete)...",
                                    artifact_name,
                                    pct
                                );
                            } else {
                                n!(
                                    "download_progress",
                                    "⏳ Still downloading {}...",
                                    artifact_name
                                );
                            }
                        }
                    }
                }
            }).await
        })
    }

    /// Cancel the download
    pub fn cancel(&self) {
        self.cancel_token.cancel();
    }

    /// Check if cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.is_cancelled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_progress_tracking() {
        let cancel_token = CancellationToken::new();
        let (tracker, mut rx) = DownloadTracker::new("job-123".to_string(), Some(1000), cancel_token);

        tracker.update_progress(500);
        let progress = rx.borrow_and_update().clone();

        assert_eq!(progress.bytes_downloaded, 500);
        assert_eq!(progress.total_size, Some(1000));
        assert_eq!(progress.percentage, Some(50.0));
    }

    #[tokio::test]
    async fn test_cancellation() {
        let cancel_token = CancellationToken::new();
        let (tracker, _rx) = DownloadTracker::new("job-123".to_string(), None, cancel_token);

        assert!(!tracker.is_cancelled());
        tracker.cancel();
        assert!(tracker.is_cancelled());
    }

    #[tokio::test]
    async fn test_increment_progress() {
        let cancel_token = CancellationToken::new();
        let (tracker, mut rx) = DownloadTracker::new("job-123".to_string(), Some(1000), cancel_token);

        tracker.increment_progress(300);
        tracker.increment_progress(200);

        let progress = rx.borrow_and_update().clone();
        assert_eq!(progress.bytes_downloaded, 500);
    }
}
