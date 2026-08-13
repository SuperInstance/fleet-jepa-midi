// src/director/gateway.rs — fleet-gateway API client
//
// Connects to the fleet-gateway service to route LLM calls
// with circuit breaker and model routing.

use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use super::PhrasingCall;
use crate::jepa::embedding::Embedding;

/// HTTP client for the fleet-gateway API.
pub struct GatewayClient {
    base_url: String,
    client: Client,
}

impl GatewayClient {
    pub fn new(base_url: &str) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_millis(1500)) // 1500ms max per design doc
            .build()
            .expect("failed to build HTTP client");

        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client,
        }
    }

    /// Send a phrasing request to the LLM bandleader via fleet-gateway.
    ///
    /// The request includes the current JEPA embedding (translated to
    /// sensory context) and directive history. The response is a
    /// structured phrasing directive.
    pub async fn request_directive(
        &self,
        embedding: &Embedding,
        context: &SensoryContext,
    ) -> anyhow::Result<PhrasingCall> {
        let payload = GatewayRequest {
            embedding: embedding.to_vec(),
            context: context.clone(),
        };

        let resp = self.client
            .post(format!("{}/v1/jepa/directive", self.base_url))
            .json(&payload)
            .send()
            .await?
            .error_for_status()?
            .json::<PhrasingCall>()
            .await?;

        Ok(resp)
    }

    /// Health check.
    pub async fn health(&self) -> anyhow::Result<bool> {
        let resp = self.client
            .get(format!("{}/health", self.base_url))
            .send()
            .await?;

        Ok(resp.status().is_success())
    }
}

/// Sensory context sent to the LLM alongside the embedding.
///
/// This is the "embedding-to-text bridge" from the design doc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensoryContext {
    pub bar_number: u32,
    pub form_position: String,
    pub key: String,
    pub tempo: f32,
    pub style: String,
    pub bars_since_last_directive: u32,
    pub last_directive_status: String,
}

impl Default for SensoryContext {
    fn default() -> Self {
        Self {
            bar_number: 1,
            form_position: "A1".to_string(),
            key: "C major".to_string(),
            tempo: 120.0,
            style: "medium swing".to_string(),
            bars_since_last_directive: 4,
            last_directive_status: "completed".to_string(),
        }
    }
}

/// Request payload to fleet-gateway.
#[derive(Debug, Serialize)]
struct GatewayRequest {
    embedding: Vec<f32>,
    context: SensoryContext,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_construction() {
        let client = GatewayClient::new("http://localhost:3000/");
        assert_eq!(client.base_url, "http://localhost:3000");
    }

    #[test]
    fn test_default_context() {
        let ctx = SensoryContext::default();
        assert_eq!(ctx.bar_number, 1);
        assert_eq!(ctx.tempo, 120.0);
    }

    #[tokio::test]
    async fn test_health_check_unreachable() {
        // Nobody listening on this port — should fail gracefully
        let client = GatewayClient::new("http://127.0.0.1:1");
        let result = client.health().await;
        assert!(result.is_err() || !result.unwrap());
    }
}
