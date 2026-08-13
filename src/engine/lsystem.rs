// src/engine/lsystem.rs — L-system pattern generator
//
// A simple Lindenmayer system with musical production rules.
// The L-system generates a string of symbols, which are then
// mapped to musical intervals and durations.

use std::collections::HashMap;
use crate::engine::{AlgorithmicEngine, NoteSeq};
use crate::midi::MidiNote;

/// An L-system rule set for music.
pub struct LSystem {
    /// Axiom (starting string).
    axiom: Vec<char>,
    /// Production rules: symbol → replacement string.
    rules: HashMap<char, Vec<char>>,
    /// Number of iterations to expand.
    iterations: usize,
}

impl LSystem {
    pub fn new() -> Self {
        let mut rules = HashMap::new();
        // Musical L-system rules:
        // F = go up a step
        // G = go up a third
        // H = go down a step
        // + = rest
        // X, Y = recursive expansion symbols
        rules.insert('F', vec!['F', 'G', 'X']);
        rules.insert('G', vec!['F', 'H']);
        rules.insert('X', vec!['G', '+', 'Y']);
        rules.insert('Y', vec!['H', 'F']);

        Self {
            axiom: vec!['F', 'X'],
            rules,
            iterations: 3,
        }
    }

    /// Expand the L-system for `iterations` rounds.
    pub fn expand(&self) -> Vec<char> {
        let mut current = self.axiom.clone();
        for _ in 0..self.iterations {
            let mut next = Vec::new();
            for &c in &current {
                match self.rules.get(&c) {
                    Some(replacement) => next.extend(replacement),
                    None => next.push(c),
                }
            }
            current = next;
        }
        current
    }

    /// Map expanded symbols to MIDI notes.
    ///
    /// F: up a whole step (+2)
    /// G: up a minor third (+3)
    /// H: down a whole step (-2)
    /// +: rest (skip a beat)
    /// Other symbols are ignored.
    pub fn to_notes(&self, symbols: &[char], base_pitch: u8) -> Vec<MidiNote> {
        let mut notes = Vec::new();
        let mut pitch = base_pitch as i32;
        let mut beat = 0.0f32;

        for &c in symbols {
            match c {
                'F' => {
                    pitch += 2;
                    notes.push(MidiNote {
                        pitch: pitch.clamp(0, 127) as u8,
                        start_beat: beat,
                        duration_beats: 0.5,
                        velocity: 80,
                    });
                    beat += 0.5;
                }
                'G' => {
                    pitch += 3;
                    notes.push(MidiNote {
                        pitch: pitch.clamp(0, 127) as u8,
                        start_beat: beat,
                        duration_beats: 0.5,
                        velocity: 90,
                    });
                    beat += 0.5;
                }
                'H' => {
                    pitch -= 2;
                    notes.push(MidiNote {
                        pitch: pitch.clamp(0, 127) as u8,
                        start_beat: beat,
                        duration_beats: 0.5,
                        velocity: 70,
                    });
                    beat += 0.5;
                }
                '+' => {
                    // Rest — just advance time
                    beat += 0.5;
                }
                _ => {} // ignore expansion symbols
            }
        }

        notes
    }
}

impl Default for LSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// L-system engine wrapper.
pub struct LSystemEngine {
    lsystem: LSystem,
}

impl Default for LSystemEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl LSystemEngine {
    pub fn new() -> Self {
        Self { lsystem: LSystem::new() }
    }
}

impl AlgorithmicEngine for LSystemEngine {
    fn name(&self) -> &str { "lsystem" }

    fn generate(&self, bars: usize, _tempo: f32) -> NoteSeq {
        let symbols = self.lsystem.expand();
        let notes = self.lsystem.to_notes(&symbols, 60);

        // Truncate or pad to the requested number of bars
        let max_beats = bars as f32 * 4.0;
        notes.into_iter()
            .filter(|n| n.start_beat < max_beats)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand() {
        let ls = LSystem::new();
        let expanded = ls.expand();
        // With 3 iterations starting from FX, we should have a substantial string
        assert!(expanded.len() > 4, "expanded string should be longer than axiom");
    }

    #[test]
    fn test_to_notes() {
        let ls = LSystem::new();
        let symbols = ls.expand();
        let notes = ls.to_notes(&symbols, 60);
        assert!(!notes.is_empty());
        for n in &notes {
            assert!(n.pitch <= 127);
            assert!(n.duration_beats > 0.0);
        }
    }

    #[test]
    fn test_engine_generate() {
        let engine = LSystemEngine::new();
        let notes = engine.generate(4, 120.0);
        assert!(!notes.is_empty());
        // All notes should be within 4 bars
        for n in &notes {
            assert!(n.start_beat < 16.0);
        }
    }

    #[test]
    fn test_deterministic() {
        let engine = LSystemEngine::new();
        let notes1 = engine.generate(4, 120.0);
        let notes2 = engine.generate(4, 120.0);
        assert_eq!(notes1, notes2, "L-system should be deterministic");
    }

    #[test]
    fn test_rest_advances_time() {
        let ls = LSystem::new();
        let notes = ls.to_notes(&['F', '+', 'F'], 60);
        // Two notes with a rest between them — second note should be at beat 1.0
        assert_eq!(notes.len(), 2);
        assert_approx_eq::assert_approx_eq!(notes[0].start_beat, 0.0, 0.001);
        assert_approx_eq::assert_approx_eq!(notes[1].start_beat, 1.0, 0.001);
    }
}
