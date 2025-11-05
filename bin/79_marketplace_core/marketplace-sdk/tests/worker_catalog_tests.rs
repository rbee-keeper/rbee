// TEAM-409: Unit tests for WorkerCatalogClient
// Created by: TEAM-409

use marketplace_sdk::{
    WorkerCatalogClient, WorkerFilter,
    WorkerType, Platform, Architecture,
};

#[cfg(test)]
mod worker_catalog_tests {
    use super::*;

    #[test]
    fn test_worker_filter_default() {
        let filter = WorkerFilter::default();
        assert!(filter.worker_type.is_none());
        assert!(filter.platform.is_none());
        assert!(filter.architecture.is_none());
        assert!(filter.min_context_length.is_none());
        assert!(filter.model_architecture.is_none());
        assert!(filter.model_format.is_none());
    }

    #[test]
    fn test_worker_filter_with_values() {
        let filter = WorkerFilter {
            worker_type: Some(WorkerType::Cuda),
            platform: Some(Platform::Linux),
            architecture: Some(Architecture::X86_64),
            min_context_length: Some(8192),
            model_architecture: Some("llama".to_string()),
            model_format: Some("safetensors".to_string()),
        };

        assert_eq!(filter.worker_type, Some(WorkerType::Cuda));
        assert_eq!(filter.platform, Some(Platform::Linux));
        assert_eq!(filter.architecture, Some(Architecture::X86_64));
        assert_eq!(filter.min_context_length, Some(8192));
        assert_eq!(filter.model_architecture, Some("llama".to_string()));
        assert_eq!(filter.model_format, Some("safetensors".to_string()));
    }

    #[test]
    fn test_client_creation() {
        let client = WorkerCatalogClient::new("http://localhost:3000");
        // Client should be created successfully
        // We can't test much without making actual HTTP requests
        drop(client);
    }

    #[test]
    fn test_default_client() {
        let client = WorkerCatalogClient::default();
        // Default client should use localhost:3000
        drop(client);
    }

    #[test]
    fn test_architecture_enum() {
        let x86 = Architecture::X86_64;
        let arm = Architecture::Aarch64;

        assert_eq!(x86.as_str(), "x86_64");
        assert_eq!(arm.as_str(), "aarch64");
    }

    #[test]
    fn test_architecture_display() {
        let x86 = Architecture::X86_64;
        assert_eq!(format!("{}", x86), "x86_64");

        let arm = Architecture::Aarch64;
        assert_eq!(format!("{}", arm), "aarch64");
    }

    // NOTE: The following tests require a running Hono server
    // They are commented out to avoid test failures in CI
    // Uncomment and run manually when Hono server is available

    /*
    #[tokio::test]
    async fn test_list_workers() {
        let client = WorkerCatalogClient::default();
        let workers = client.list_workers().await.unwrap();
        assert!(!workers.is_empty(), "Should have at least one worker");
    }

    #[tokio::test]
    async fn test_get_worker() {
        let client = WorkerCatalogClient::default();
        
        // First get all workers to find a valid ID
        let workers = client.list_workers().await.unwrap();
        assert!(!workers.is_empty());
        
        let first_worker_id = &workers[0].id;
        
        // Now get that specific worker
        let worker = client.get_worker(first_worker_id).await.unwrap();
        assert!(worker.is_some());
        assert_eq!(worker.unwrap().id, *first_worker_id);
    }

    #[tokio::test]
    async fn test_get_nonexistent_worker() {
        let client = WorkerCatalogClient::default();
        let worker = client.get_worker("nonexistent-worker-id").await.unwrap();
        assert!(worker.is_none());
    }

    #[tokio::test]
    async fn test_filter_by_type() {
        let client = WorkerCatalogClient::default();
        let filter = WorkerFilter {
            worker_type: Some(WorkerType::Cuda),
            ..Default::default()
        };
        
        let workers = client.filter_workers(filter).await.unwrap();
        
        // All returned workers should be CUDA
        for worker in workers {
            assert_eq!(worker.worker_type, WorkerType::Cuda);
        }
    }

    #[tokio::test]
    async fn test_filter_by_platform() {
        let client = WorkerCatalogClient::default();
        let filter = WorkerFilter {
            platform: Some(Platform::Linux),
            ..Default::default()
        };
        
        let workers = client.filter_workers(filter).await.unwrap();
        
        // All returned workers should support Linux
        for worker in workers {
            assert!(worker.supports_platform(Platform::Linux));
        }
    }

    #[tokio::test]
    async fn test_filter_by_architecture() {
        let client = WorkerCatalogClient::default();
        let filter = WorkerFilter {
            architecture: Some(Architecture::X86_64),
            ..Default::default()
        };
        
        let workers = client.filter_workers(filter).await.unwrap();
        
        // All returned workers should support x86_64
        for worker in workers {
            assert!(worker.supports_architecture(Architecture::X86_64));
        }
    }

    #[tokio::test]
    async fn test_filter_by_context_length() {
        let client = WorkerCatalogClient::default();
        let filter = WorkerFilter {
            min_context_length: Some(8192),
            ..Default::default()
        };
        
        let workers = client.filter_workers(filter).await.unwrap();
        
        // All returned workers should have context length >= 8192
        for worker in workers {
            if let Some(max_context) = worker.max_context_length {
                assert!(max_context >= 8192);
            }
        }
    }

    #[tokio::test]
    async fn test_find_compatible_workers() {
        let client = WorkerCatalogClient::default();
        let workers = client
            .find_compatible_workers("llama", "safetensors")
            .await
            .unwrap();
        
        // All returned workers should support llama and safetensors
        for worker in workers {
            assert!(worker.supports_format("llama") || worker.supports_format("safetensors"));
        }
    }

    #[tokio::test]
    async fn test_multiple_filters() {
        let client = WorkerCatalogClient::default();
        let filter = WorkerFilter {
            worker_type: Some(WorkerType::Cuda),
            platform: Some(Platform::Linux),
            architecture: Some(Architecture::X86_64),
            min_context_length: Some(8192),
            ..Default::default()
        };
        
        let workers = client.filter_workers(filter).await.unwrap();
        
        // All returned workers should match all criteria
        for worker in workers {
            assert_eq!(worker.worker_type, WorkerType::Cuda);
            assert!(worker.supports_platform(Platform::Linux));
            assert!(worker.supports_architecture(Architecture::X86_64));
            if let Some(max_context) = worker.max_context_length {
                assert!(max_context >= 8192);
            }
        }
    }
    */
}
