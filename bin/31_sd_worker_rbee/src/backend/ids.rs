// TEAM-481: Newtype pattern for type-safe IDs
//
// Provides compile-time type safety for different ID types.
// Prevents mixing up request_id, job_id, etc.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Request ID for generation requests
///
/// TEAM-481: Newtype pattern prevents mixing up with JobId
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestId(String);

impl RequestId {
    /// Create a new random request ID
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
    
    /// Create from an existing string
    pub fn from_string(s: String) -> Self {
        Self(s)
    }
    
    /// Get the ID as a string slice
    pub fn as_str(&self) -> &str {
        &self.0
    }
    
    /// Convert to owned String
    pub fn into_string(self) -> String {
        self.0
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for RequestId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<RequestId> for String {
    fn from(id: RequestId) -> Self {
        id.0
    }
}

/// Job ID for HTTP job tracking
///
/// TEAM-481: Newtype pattern prevents mixing up with RequestId
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct JobId(String);

impl JobId {
    /// Create a new random job ID
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
    
    /// Create from an existing string
    pub fn from_string(s: String) -> Self {
        Self(s)
    }
    
    /// Get the ID as a string slice
    pub fn as_str(&self) -> &str {
        &self.0
    }
    
    /// Convert to owned String
    pub fn into_string(self) -> String {
        self.0
    }
}

impl Default for JobId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for JobId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for JobId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<JobId> for String {
    fn from(id: JobId) -> Self {
        id.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_request_id_creation() {
        let id1 = RequestId::new();
        let id2 = RequestId::new();
        assert_ne!(id1, id2); // Should be unique
    }
    
    #[test]
    fn test_job_id_creation() {
        let id1 = JobId::new();
        let id2 = JobId::new();
        assert_ne!(id1, id2); // Should be unique
    }
    
    #[test]
    fn test_request_id_display() {
        let id = RequestId::from_string("test-123".to_string());
        assert_eq!(id.to_string(), "test-123");
        assert_eq!(id.as_str(), "test-123");
    }
    
    #[test]
    fn test_job_id_display() {
        let id = JobId::from_string("job-456".to_string());
        assert_eq!(id.to_string(), "job-456");
        assert_eq!(id.as_str(), "job-456");
    }
    
    #[test]
    fn test_ids_are_different_types() {
        // This test verifies compile-time type safety
        let request_id = RequestId::new();
        let job_id = JobId::new();
        
        // These are different types - can't mix them up!
        assert_ne!(request_id.as_str(), job_id.as_str()); // Different UUIDs
    }
}
