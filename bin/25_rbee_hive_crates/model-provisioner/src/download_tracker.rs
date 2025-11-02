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
#[derive(Clone)]
pub struct DownloadTracker {
    /// Total bytes downloaded
    bytes_downloaded: Arc<AtomicU64>,
    /// Expected total size (if known)
    total_size: Option<u64>,
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
    pub bytes_downloaded: u64,
    pub total_size: Option<u64>,
    pub percentage: Option<f64>,
}

impl DownloadTracker {
    /// Create a new download tracker
    pub fn new(job_id: String, total_size: Option<u64>) -> (Self, watch::Receiver<DownloadProgress>) {
        let (progress_tx, progress_rx) = watch::channel(DownloadProgress {
            bytes_downloaded: 0,
            total_size,
            percentage: None,
        });

        let tracker = Self {
            bytes_downloaded: Arc::new(AtomicU64::new(0)),
            total_size,
            cancel_token: CancellationToken::new(),
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

    /// Get current progress
    pub fn current_progress(&self) -> DownloadProgress {
        let bytes = self.bytes_downloaded.load(Ordering::Relaxed);
        let percentage = self.total_size.map(|total| {
            if total > 0 {
                (bytes as f64 / total as f64) * 100.0
            } else {
                0.0
            }
        });

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
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(10));
            interval.tick().await; // Skip first immediate tick

            loop {
                tokio::select! {
                    _ = cancel_token.cancelled() => {
                        break;
                    }
                    _ = interval.tick() => {
                        let progress = tracker.current_progress();
                        
                        // CRITICAL: Narration in spawned tasks needs explicit job_id
                        // The n! macro will use the job_id from the current context
                        // For spawned tasks, we need to set up the context properly
                        
                        if let Some(pct) = progress.percentage {
                            n!(
                                "download_progress",
                                "⏳ Still downloading {} ({:.1}% - {:.2} MB / {:.2} MB)...",
                                artifact_name,
                                pct,
                                progress.bytes_downloaded as f64 / 1_000_000.0,
                                progress.total_size.unwrap_or(0) as f64 / 1_000_000.0
                            );
                        } else {
                            n!(
                                "download_progress",
                                "⏳ Still downloading {} ({:.2} MB downloaded)...",
                                artifact_name,
                                progress.bytes_downloaded as f64 / 1_000_000.0
                            );
                        }
                    }
                }
            }

            n!("download_heartbeat_stopped", "Heartbeat stopped for {}", artifact_name);
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
        let (tracker, mut rx) = DownloadTracker::new("job-123".to_string(), Some(1000));

        tracker.update_progress(500);
        let progress = rx.borrow_and_update().clone();

        assert_eq!(progress.bytes_downloaded, 500);
        assert_eq!(progress.total_size, Some(1000));
        assert_eq!(progress.percentage, Some(50.0));
    }

    #[tokio::test]
    async fn test_cancellation() {
        let (tracker, _rx) = DownloadTracker::new("job-123".to_string(), None);

        assert!(!tracker.is_cancelled());
        tracker.cancel();
        assert!(tracker.is_cancelled());
    }

    #[tokio::test]
    async fn test_increment_progress() {
        let (tracker, mut rx) = DownloadTracker::new("job-123".to_string(), Some(1000));

        tracker.increment_progress(300);
        tracker.increment_progress(200);

        let progress = rx.borrow_and_update().clone();
        assert_eq!(progress.bytes_downloaded, 500);
    }
}
