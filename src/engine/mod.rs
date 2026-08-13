// src/engine/mod.rs — Algorithmic engine registry

pub mod markov;
pub mod lsystem;
pub mod ca;
pub mod fractal;

use crate::midi::MidiNote;

/// A note sequence produced by an engine.
pub type NoteSeq = Vec<MidiNote>;

/// Trait for all algorithmic engines.
pub trait AlgorithmicEngine: Send + Sync {
    /// Engine name (e.g., "markov", "lsystem").
    fn name(&self) -> &str;
    /// Generate `bars` bars of music at the given tempo.
    fn generate(&self, bars: usize, tempo: f32) -> NoteSeq;
}

/// Registry of available engines.
pub struct EngineRegistry {
    engines: Vec<Box<dyn AlgorithmicEngine>>,
}

impl EngineRegistry {
    pub fn new() -> Self {
        let engines: Vec<Box<dyn AlgorithmicEngine>> = vec![
            Box::new(markov::MarkovEngine::new()),
            Box::new(lsystem::LSystemEngine::new()),
            Box::new(ca::CaEngine::new()),
            Box::new(fractal::FractalEngine::new()),
        ];
        Self { engines }
    }

    /// List available engine names.
    pub fn names(&self) -> Vec<&str> {
        self.engines.iter().map(|e| e.name()).collect()
    }

    /// Generate notes with the named engine.
    pub fn generate(&self, name: &str, bars: usize) -> anyhow::Result<NoteSeq> {
        for engine in &self.engines {
            if engine.name() == name {
                return Ok(engine.generate(bars, 120.0));
            }
        }
        anyhow::bail!("unknown engine: '{name}'. Available: {:?}", self.names())
    }
}

impl Default for EngineRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_has_all_engines() {
        let reg = EngineRegistry::new();
        let names = reg.names();
        assert!(names.contains(&"markov"));
        assert!(names.contains(&"lsystem"));
        assert!(names.contains(&"ca"));
        assert!(names.contains(&"fractal"));
    }

    #[test]
    fn test_unknown_engine_errors() {
        let reg = EngineRegistry::new();
        assert!(reg.generate("nonexistent", 4).is_err());
    }

    #[test]
    fn test_each_engine_generates_notes() {
        let reg = EngineRegistry::new();
        for name in reg.names() {
            let notes = reg.generate(name, 4).unwrap();
            assert!(!notes.is_empty(), "engine '{name}' produced no notes");
        }
    }
}
