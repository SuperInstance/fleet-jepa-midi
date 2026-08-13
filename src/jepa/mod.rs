// src/jepa/mod.rs — JEPA encoder interface

pub mod embedding;
pub mod predictor;

pub use embedding::{JepaEncoder, BarFeatures, EMBEDDING_DIM};
pub use predictor::LinearPredictor;
