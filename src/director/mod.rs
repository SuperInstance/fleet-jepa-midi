// src/director/mod.rs — LLM bandleader interface

pub mod phrasing;
pub mod gateway;

pub use phrasing::{Directive, DirectiveAction, PhrasingCall, ScalarTarget, ScalarMode, Priority};
pub use gateway::GatewayClient;

use crate::jepa::JepaEncoder;
use crate::midi::Bar;

/// The Director is the LLM bandleader. It reads JEPA embeddings,
/// decides direction via fleet-gateway, and outputs phrasing directives.
pub struct Director {
    gateway: GatewayClient,
    encoder: JepaEncoder,
}

impl Director {
    pub fn new(gateway_url: &str) -> Self {
        Self {
            gateway: GatewayClient::new(gateway_url),
            encoder: JepaEncoder::new(),
        }
    }

    /// Run the director loop. In v1 this is a stub that logs state.
    pub async fn run(&mut self) -> anyhow::Result<()> {
        tracing::info!("director running — v1 stub mode");

        let bar = Bar::default();
        let embedding = self.encoder.embed_bar(&bar);
        tracing::info!("initial embedding: {:?}", embedding);

        // Try health check on gateway
        match self.gateway.health().await {
            Ok(true) => tracing::info!("gateway is healthy"),
            Ok(false) => tracing::warn!("gateway returned unhealthy status"),
            Err(e) => tracing::warn!("gateway unreachable: {e}"),
        }

        tracing::info!("director stub complete — implement full loop in v2");
        Ok(())
    }
}
