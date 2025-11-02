// TEAM-135: Created by TEAM-135 (scaffolding)
// TEAM-385: Implemented basic progress tracking with narration
// Purpose: Download progress tracking for model downloads

#![warn(missing_docs)]
#![warn(clippy::all)]

//! rbee-hive-download-tracker
//!
//! Download progress tracking for model downloads
//!
//! Provides progress callbacks for tracking download operations.
//! Integrates with narration system for real-time progress updates.

use observability_narration_core::n;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Progress tracker for downloads
///
/// Tracks bytes downloaded and provides progress updates via narration.
///
/// # Example
///
/// ```rust,no_run
/// use rbee_hive_download_tracker::ProgressTracker;
///
/// let tracker = ProgressTracker::new("model-123", 1_000_000_000); // 1GB
///
/// // Update progress as download proceeds
/// tracker.update(100_000_000); // 100MB
/// tracker.update(500_000_000); // 500MB
/// tracker.complete();
/// ```
pub struct ProgressTracker {
    download_id: String,
    total_bytes: u64,
    bytes_downloaded: Arc<AtomicU64>,
    start_time: Instant,
    last_update: Arc<std::sync::Mutex<Instant>>,
}

impl ProgressTracker {
    /// Create a new progress tracker
    ///
    /// # Arguments
    /// * `download_id` - Unique identifier for this download (e.g., model ID)
    /// * `total_bytes` - Total size of download in bytes
    pub fn new(download_id: impl Into<String>, total_bytes: u64) -> Self {
        let download_id = download_id.into();
        n!("download_start", "📥 Starting download: {} ({:.2} GB)", download_id, total_bytes as f64 / 1_000_000_000.0);
        
        Self {
            download_id,
            total_bytes,
            bytes_downloaded: Arc::new(AtomicU64::new(0)),
            start_time: Instant::now(),
            last_update: Arc::new(std::sync::Mutex::new(Instant::now())),
        }
    }
    
    /// Update progress with new byte count
    ///
    /// Emits progress narration every 5 seconds or 100MB, whichever comes first.
    pub fn update(&self, bytes: u64) {
        self.bytes_downloaded.store(bytes, Ordering::Relaxed);
        
        // Check if we should emit progress update
        let mut last = self.last_update.lock().unwrap();
        let now = Instant::now();
        let elapsed_since_update = now.duration_since(*last);
        let bytes_since_start = bytes;
        
        // Emit progress every 5 seconds or every 100MB
        let should_update = elapsed_since_update >= Duration::from_secs(5) 
            || (bytes_since_start > 0 && bytes_since_start % 100_000_000 < 10_000_000); // ~100MB chunks
        
        if should_update {
            let percent = (bytes as f64 / self.total_bytes as f64) * 100.0;
            let elapsed = self.start_time.elapsed();
            let speed_bps = if elapsed.as_secs() > 0 {
                bytes / elapsed.as_secs()
            } else {
                0
            };
            let speed_mbps = speed_bps as f64 / 1_000_000.0;
            
            n!(
                "download_progress",
                "📊 Progress: {:.1}% ({:.2} GB / {:.2} GB) @ {:.1} MB/s",
                percent,
                bytes as f64 / 1_000_000_000.0,
                self.total_bytes as f64 / 1_000_000_000.0,
                speed_mbps
            );
            
            *last = now;
        }
    }
    
    /// Mark download as complete
    pub fn complete(&self) {
        let elapsed = self.start_time.elapsed();
        let bytes = self.bytes_downloaded.load(Ordering::Relaxed);
        let speed_mbps = if elapsed.as_secs() > 0 {
            (bytes / elapsed.as_secs()) as f64 / 1_000_000.0
        } else {
            0.0
        };
        
        n!(
            "download_complete",
            "✅ Download complete: {} ({:.2} GB in {:.1}s @ {:.1} MB/s)",
            self.download_id,
            bytes as f64 / 1_000_000_000.0,
            elapsed.as_secs_f64(),
            speed_mbps
        );
    }
    
    /// Mark download as failed
    pub fn fail(&self, error: &str) {
        let bytes = self.bytes_downloaded.load(Ordering::Relaxed);
        n!(
            "download_failed",
            "❌ Download failed: {} ({:.2} GB downloaded) - {}",
            self.download_id,
            bytes as f64 / 1_000_000_000.0,
            error
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tracker_creation() {
        let tracker = ProgressTracker::new("test-model", 1_000_000_000);
        assert_eq!(tracker.total_bytes, 1_000_000_000);
        assert_eq!(tracker.bytes_downloaded.load(Ordering::Relaxed), 0);
    }
    
    #[test]
    fn test_progress_update() {
        let tracker = ProgressTracker::new("test-model", 1_000_000_000);
        tracker.update(500_000_000);
        assert_eq!(tracker.bytes_downloaded.load(Ordering::Relaxed), 500_000_000);
    }
}
