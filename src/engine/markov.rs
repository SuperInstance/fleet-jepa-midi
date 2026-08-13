// src/engine/markov.rs — Markov chain melody generator
//
// 2nd-order Markov chain on pitch intervals.
// Trains from a melody (sequence of pitches), then generates
// new melodies by walking the chain.

use std::collections::HashMap;
use crate::engine::{AlgorithmicEngine, NoteSeq};
use crate::midi::MidiNote;

/// A 2nd-order Markov chain on pitch intervals.
pub struct MarkovChain {
    /// Map: (prev_interval, curr_interval) → list of next intervals.
    transitions: HashMap<(i32, i32), Vec<i32>>,
    /// Starting pitch for generation.
    base_pitch: u8,
}

impl MarkovChain {
    pub fn new() -> Self {
        Self {
            transitions: HashMap::new(),
            base_pitch: 60, // C4
        }
    }

    /// Train the chain from a melody (sequence of MIDI pitches).
    pub fn train(&mut self, pitches: &[u8]) {
        if pitches.len() < 4 {
            return;
        }

        // Compute intervals
        let intervals: Vec<i32> = pitches.windows(2)
            .map(|w| w[1] as i32 - w[0] as i32)
            .collect();

        // Build 2nd-order transitions
        for w in intervals.windows(3) {
            let key = (w[0], w[1]);
            self.transitions.entry(key)
                .or_default()
                .push(w[2]);
        }

        // Set base pitch to the first note
        self.base_pitch = pitches[0];
    }

    /// Generate a sequence of pitches by walking the chain.
    ///
    /// `n_notes` is the desired length. Falls back to random intervals
    /// when the chain has no transition for the current state.
    pub fn generate(&self, n_notes: usize) -> Vec<u8> {
        if n_notes == 0 {
            return vec![];
        }

        let mut result = vec![self.base_pitch];

        // Pick a starting interval pair from known transitions
        let mut prev_interval = 2i32; // whole step up
        let mut curr_interval = -1i32; // half step down

        // Try to find a real starting state
        if let Some(key) = self.transitions.keys().next() {
            prev_interval = key.0;
            curr_interval = key.1;
            let next_pitch = result[0] as i32 + prev_interval;
            result.push(next_pitch.clamp(0, 127) as u8);
            let next_pitch2 = next_pitch + curr_interval;
            result.push(next_pitch2.clamp(0, 127) as u8);
        } else {
            // No training data — generate a simple scale
            for i in 1..n_notes {
                let p = (self.base_pitch as i32 + i as i32 * 2).clamp(0, 127) as u8;
                result.push(p);
            }
            return result;
        }

        while result.len() < n_notes {
            let key = (prev_interval, curr_interval);
            if let Some(next_intervals) = self.transitions.get(&key) {
                // Pick a random next interval (deterministic for reproducibility — pick middle)
                let idx = next_intervals.len() / 2;
                let next_int = next_intervals[idx];
                let last_pitch = *result.last().unwrap() as i32;
                let new_pitch = (last_pitch + next_int).clamp(0, 127) as u8;
                result.push(new_pitch);
                prev_interval = curr_interval;
                curr_interval = next_int;
            } else {
                // Unknown state: pick a small step to keep going
                let fallback = match result.len() % 3 {
                    0 => 2,
                    1 => -1,
                    _ => 0,
                };
                let last_pitch = *result.last().unwrap() as i32;
                let new_pitch = (last_pitch + fallback).clamp(0, 127) as u8;
                result.push(new_pitch);
                prev_interval = curr_interval;
                curr_interval = fallback;
            }
        }

        result.truncate(n_notes);
        result
    }
}

impl Default for MarkovChain {
    fn default() -> Self {
        Self::new()
    }
}

/// Markov chain engine wrapper.
pub struct MarkovEngine {
    chain: MarkovChain,
}

impl MarkovEngine {
    pub fn new() -> Self {
        let mut chain = MarkovChain::new();
        // Pre-train with a simple C major scale melody
        chain.train(&[60, 62, 64, 65, 67, 69, 71, 72,
                       71, 69, 67, 65, 64, 62, 60,
                       62, 64, 65, 67, 67, 65, 64, 62]);
        Self { chain }
    }
}

impl AlgorithmicEngine for MarkovEngine {
    fn name(&self) -> &str { "markov" }

    fn generate(&self, bars: usize, tempo: f32) -> NoteSeq {
        let beats_per_bar = 4usize;
        let notes_per_bar = 8; // eighth notes
        let total_notes = bars * notes_per_bar;
        let pitches = self.chain.generate(total_notes);

        let mut notes = Vec::with_capacity(total_notes);
        for (i, &pitch) in pitches.iter().enumerate() {
            let bar = i / notes_per_bar;
            let step_in_bar = i % notes_per_bar;
            let start_beat = bar as f32 * beats_per_bar as f32 + step_in_bar as f32 * 0.5;
            notes.push(MidiNote {
                pitch,
                start_beat,
                duration_beats: 0.5,
                velocity: 80 + (i % 3) * 10,
            });
        }

        // unused but kept for future tempo-aware timing
        let _ = tempo;
        notes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_and_generate() {
        let mut chain = MarkovChain::new();
        // C major scale up and down
        chain.train(&[60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62, 60]);
        let generated = chain.generate(20);
        assert_eq!(generated.len(), 20);
        // All pitches should be valid MIDI
        for &p in &generated {
            assert!(p <= 127);
        }
    }

    #[test]
    fn test_empty_training() {
        let chain = MarkovChain::new();
        let generated = chain.generate(10);
        assert_eq!(generated.len(), 10);
        // Should still produce valid pitches
        for &p in &generated {
            assert!(p <= 127);
        }
    }

    #[test]
    fn test_engine_generate() {
        let engine = MarkovEngine::new();
        let notes = engine.generate(4, 120.0);
        assert!(!notes.is_empty());
        assert_eq!(notes.len(), 32); // 4 bars × 8 eighth notes
        for n in &notes {
            assert!(n.pitch <= 127);
            assert!(n.velocity > 0);
        }
    }

    #[test]
    fn test_engine_name() {
        let engine = MarkovEngine::new();
        assert_eq!(engine.name(), "markov");
    }
}
