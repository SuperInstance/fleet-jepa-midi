// src/engine/fractal.rs — Fractal parameter generator
//
// Uses iterated function systems (IFS) and Mandelbrot-style
// escape-time calculations to generate musical parameters.
// The fractal landscape is traversed to produce evolving
// pitch, velocity, and duration contours.

use crate::engine::{AlgorithmicEngine, NoteSeq};
use crate::midi::MidiNote;

/// A fractal melody generator using the Mandelbrot set escape-time
/// function mapped onto musical dimensions.
pub struct FractalGenerator {
    /// Center of the Mandelbrot exploration (real axis).
    cx: f64,
    /// Center of the Mandelbrot exploration (imaginary axis).
    cy: f64,
    /// Scale of the exploration window.
    scale: f64,
}

impl FractalGenerator {
    pub fn new() -> Self {
        Self {
            cx: -0.7,
            cy: 0.27015, // classic Julia set position
            scale: 0.02,
        }
    }

    /// Compute the Mandelbrot/Julia escape time at (x, y).
    /// Returns the iteration count before the point escapes (or max_iter).
    fn escape_time(&self, x: f64, y: f64, max_iter: u32) -> u32 {
        let mut zx = x;
        let mut zy = y;
        let mut i = 0;
        while i < max_iter && zx * zx + zy * zy < 4.0 {
            let tmp = zx * zx - zy * zy + self.cx;
            zy = 2.0 * zx * zy + self.cy;
            zx = tmp;
            i += 1;
        }
        i
    }

    /// Generate a sequence of fractal-derived parameters.
    ///
    /// Walks along a line through the Julia set, sampling escape times.
    /// Each sample maps to (pitch, velocity, duration).
    pub fn generate_params(&self, n_notes: usize) -> Vec<(u8, u8, f32)> {
        let mut params = Vec::with_capacity(n_notes);
        for i in 0..n_notes {
            let t = i as f64 / n_notes as f64;
            // Walk along a diagonal through the Julia set
            let x = -1.5 + t * 3.0;
            let y = -1.0 + (t * 2.0).sin() * 0.5;

            let escape = self.escape_time(x, y, 64);

            // Map escape time to pitch (pentatonic scale)
            let scale = [0, 2, 4, 7, 9, 12, 14, 16, 19, 21]; // C pentatonic across 2 octaves
            let scale_idx = (escape as usize) % scale.len();
            let pitch = 60 + scale[scale_idx]; // base C4 + scale offset

            // Map escape time to velocity (higher escape = louder)
            let velocity = 50 + ((escape * 2) % 77) as u8;

            // Map escape time to duration
            let duration = match escape % 4 {
                0 => 0.25, // sixteenth
                1 => 0.5,  // eighth
                2 => 1.0,  // quarter
                _ => 0.75, // dotted eighth
            };

            params.push((pitch, velocity, duration));
        }
        params
    }
}

impl Default for FractalGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Fractal engine wrapper.
pub struct FractalEngine {
    generator: FractalGenerator,
}

impl FractalEngine {
    pub fn new() -> Self {
        Self { generator: FractalGenerator::new() }
    }
}

impl Default for FractalEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl AlgorithmicEngine for FractalEngine {
    fn name(&self) -> &str { "fractal" }

    fn generate(&self, bars: usize, _tempo: f32) -> NoteSeq {
        let beats_per_bar = 4.0f32;
        // Aim for roughly 8 notes per bar
        let n_notes = bars * 8;
        let params = self.generator.generate_params(n_notes);

        let mut notes = Vec::with_capacity(n_notes);
        let mut beat = 0.0f32;
        let max_beat = bars as f32 * beats_per_bar;

        for (pitch, velocity, duration) in params {
            if beat >= max_beat {
                break;
            }
            notes.push(MidiNote {
                pitch,
                start_beat: beat,
                duration_beats: duration,
                velocity,
            });
            beat += duration;
        }

        notes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_time_interior() {
        let fg = FractalGenerator::new();
        // Points deep inside the set should reach max iterations
        let escape = fg.escape_time(0.0, 0.0, 64);
        assert_eq!(escape, 64, "origin should not escape");
    }

    #[test]
    fn test_escape_time_exterior() {
        let fg = FractalGenerator::new();
        // Points far from the set should escape quickly
        let escape = fg.escape_time(10.0, 10.0, 64);
        assert!(escape < 5, "far point should escape quickly");
    }

    #[test]
    fn test_generate_params() {
        let fg = FractalGenerator::new();
        let params = fg.generate_params(32);
        assert_eq!(params.len(), 32);
        for (pitch, vel, dur) in &params {
            assert!(*pitch <= 127);
            assert!(*vel > 0);
            assert!(*dur > 0.0);
        }
    }

    #[test]
    fn test_engine_generate() {
        let engine = FractalEngine::new();
        let notes = engine.generate(4, 120.0);
        assert!(!notes.is_empty());
        for n in &notes {
            assert!(n.start_beat < 16.0);
            assert!(n.pitch <= 127);
        }
    }

    #[test]
    fn test_engine_name() {
        let engine = FractalEngine::new();
        assert_eq!(engine.name(), "fractal");
    }

    #[test]
    fn test_deterministic() {
        let engine = FractalEngine::new();
        let notes1 = engine.generate(4, 120.0);
        let notes2 = engine.generate(4, 120.0);
        assert_eq!(notes1, notes2, "fractal should be deterministic");
    }
}
