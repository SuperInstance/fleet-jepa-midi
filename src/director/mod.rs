// src/director/mod.rs — LLM bandleader interface

pub mod phrasing;
pub mod gateway;

pub use phrasing::{Directive, DirectiveAction, PhrasingCall, ScalarTarget, ScalarMode, Priority};
pub use gateway::GatewayClient;
