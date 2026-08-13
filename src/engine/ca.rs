// src/engine/ca.rs — Cellular automaton rhythm generator
//
// 1D Wolfram elementary automaton used as a rhythm generator.
// Each column of the automaton maps to a time step; active cells
// trigger note events.

use crate::engine::{AlgorithmicEngine, NoteSeq};
use crate::midi::MidiNote;

/// Wolfram rule number (0-255). Determines the update rule.
const DEFAULT_RULE: u8 = 30; // Rule 30: chaotic, good for rhythms

/// A 1D elementary cellular automaton.
pub struct CellularAutomaton {
    /// Current state: one cell per time step.
    state: Vec<bool>,
    /// Wolfram rule number.
    rule: u8,
}

impl CellularAutomaton {
    pub fn new(width: usize, rule: u8) -> Self {
        let mut state = vec![false; width];
        // Seed: single active cell in the center
        state[width / 2] = true;
        Self { state, rule }
    }

    /// Advance one generation.
    pub fn step(&mut self) {
        let width = self.state.len();
        let mut next = vec![false; width];
        for i in 0..width {
            // Neighborhood: wrap around (toroidal)
            let left = self.state[(i + width - 1) % width];
            let center = self.state[i];
            let right = self.state[(i + 1) % width];
            let pattern = (left as u8) << 2 | (center as u8) << 1 | (right as u8);
            next[i] = (self.rule >> pattern) & 1 != 0;
        }
        self.state = next;
    }

    /// Get the current state as a boolean vector.
    pub fn state(&self) -> &[bool] {
        &self.state
    }

    /// Count active cells.
    pub fn active_count(&self) -> usize {
        self.state.iter().filter(|&&c| c).count()
    }
}

/// CA engine wrapper — generates rhythms from automaton patterns.
pub struct CaEngine {
    rule: u8,
}

impl CaEngine {
    pub fn new() -> Self {
        Self { rule: DEFAULT_RULE }
    }
}

impl Default for CaEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl AlgorithmicEngine for CaEngine {
    fn name(&self) -> &str { "ca" }

    fn generate(&self, bars: usize, _tempo: f32) -> NoteSeq {
        let steps_per_bar = 16; // 16th notes
        let total_steps = bars * steps_per_bar;
        let width = steps_per_bar;

        let mut ca = CellularAutomaton::new(width, self.rule);
        let mut notes = Vec::new();

        for bar in 0..bars {
            // Use current CA state as rhythm pattern
            for (step, &active) in ca.state().iter().enumerate() {
                if active {
                    let beat = bar as f32 * 4.0 + step as f32 * 0.25;
                    // Vary pitch based on step position in a pentatonic scale
                    let scale = [60, 62, 64, 67, 69, 72, 74, 76]; // C pentatonic + octave
                    let pitch = scale[step % scale.len()];
                    let velocity = 60 + (step % 4) * 15;
                    notes.push(MidiNote {
                        pitch,
                        start_beat: beat,
                        duration_beats: 0.25,
                        velocity,
                    });
                }
            }
            // Advance the CA for the next bar
            ca.step();
        }

        notes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ca_initial_state() {
        let ca = CellularAutomaton::new(16, 30);
        assert_eq!(ca.active_count(), 1); // only center cell active
    }

    #[test]
    fn test_ca_step_changes_state() {
        let mut ca = CellularAutomaton::new(16, 30);
        let initial = ca.state().to_vec();
        ca.step();
        let after = ca.state().to_vec();
        assert_ne!(initial, after, "CA should change after step");
    }

    #[test]
    fn test_ca_rule_0_all_die() {
        let mut ca = CellularAutomaton::new(16, 0);
        ca.step();
        assert_eq!(ca.active_count(), 0, "Rule 0 should kill all cells");
    }

    #[test]
    fn test_ca_rule_255_all_live() {
        let mut ca = CellularAutomaton::new(16, 255);
        ca.step();
        assert_eq!(ca.active_count(), 16, "Rule 255 should activate all cells");
    }

    #[test]
    fn test_engine_generate() {
        let engine = CaEngine::new();
        let notes = engine.generate(4, 120.0);
        assert!(!notes.is_empty());
        // Each note should be within the 4-bar range
        for n in &notes {
            assert!(n.start_beat < 16.0);
            assert!(n.pitch <= 127);
        }
    }

    #[test]
    fn test_engine_name() {
        let engine = CaEngine::new();
        assert_eq!(engine.name(), "ca");
    }
}
