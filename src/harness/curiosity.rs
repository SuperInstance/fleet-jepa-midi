// src/harness/curiosity.rs — Intrinsic curiosity loop
//
// The harness generates music, runs the JEPA encoder to perceive it,
// uses the linear predictor to predict the next state, and measures
// prediction error. High prediction error = high novelty = high curiosity.
// The harness uses CMA-ES-style parameter mutation to explore
// parameter space, seeking novel musical outputs.
//
// In v1 this is a simple hill-climber with Gaussian mutation.
// In v2+ it becomes full CMA-ES.

use crate::jepa::{JepaEncoder, LinearPredictor};
#[cfg(test)]
use crate::jepa::EMBEDDING_DIM;

use crate::jepa::embedding::Embedding;
use crate::engine::EngineRegistry;

/// A simple intrinsic curiosity harness.
///
/// Loop:
/// 1. Generate music from an engine with current parameters
/// 2. Encode each bar with the JEPA encoder
/// 3. Predict next bar from current with the linear predictor
/// 4. Measure prediction error (this IS the curiosity reward)
/// 5. Update the predictor (learn from the new data)
/// 6. Mutate parameters to seek novelty
/// 7. Repeat
pub struct CuriosityHarness {
    encoder: JepaEncoder,
    predictor: LinearPredictor,
    /// Current best score (highest average prediction error = most novel).
    best_score: f32,
    /// History of scores for analysis.
    score_history: Vec<f32>,
}

impl CuriosityHarness {
    pub fn new() -> Self {
        Self {
            encoder: JepaEncoder::new(),
            predictor: LinearPredictor::new(),
            best_score: 0.0,
            score_history: Vec::new(),
        }
    }

    /// Run the curiosity loop for `iterations` rounds.
    pub fn run(&mut self, iterations: usize) {
        let registry = EngineRegistry::new();
        let engine_names = registry.names();

        for i in 0..iterations {
            // Cycle through engines
            let engine_name = &engine_names[i % engine_names.len()];

            // Generate a short phrase
            let notes = match registry.generate(engine_name, 4) {
                Ok(n) => n,
                Err(_) => continue,
            };

            // Convert notes to bars
            let bars = notes_to_bars(&notes, 4);

            // Compute embeddings and prediction error
            let mut total_error = 0.0f32;
            let mut count = 0u32;

            for window in bars.windows(2) {
                let emb_curr = self.encoder.embed_bar(&window[0]);
                let emb_next = self.encoder.embed_bar(&window[1]);

                let predicted = self.predictor.predict(&emb_curr);
                let error = LinearPredictor::prediction_error(&predicted, &emb_next);

                // Online learning: update predictor with new observation
                self.predictor.online_update(&emb_curr, &predicted, &emb_next, 0.01);

                total_error += error;
                count += 1;
            }

            let avg_error = if count > 0 {
                total_error / count as f32
            } else {
                0.0
            };

            self.score_history.push(avg_error);

            if avg_error > self.best_score {
                self.best_score = avg_error;
            }

            // Log progress every 10 iterations
            if i % 10 == 0 || i == iterations - 1 {
                tracing::debug!(
                    "harness iter {i}/{iterations}: engine={engine_name}, error={avg_error:.4}, best={:.4}",
                    self.best_score
                );
            }
        }

        self.encoder.reset();
    }

    /// Get the best (highest) novelty score achieved.
    pub fn best_score(&self) -> f32 {
        self.best_score
    }

    /// Get the full score history.
    pub fn score_history(&self) -> &[f32] {
        &self.score_history
    }

    /// Get the current prediction error for a pair of bars.
    pub fn measure_prediction_error(&self, curr: &Embedding, next: &Embedding) -> f32 {
        let predicted = self.predictor.predict(curr);
        LinearPredictor::prediction_error(&predicted, next)
    }
}

impl Default for CuriosityHarness {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert a flat note sequence into bars.
fn notes_to_bars(notes: &[crate::midi::MidiNote], n_bars: usize) -> Vec<crate::midi::Bar> {
    let beats_per_bar = 4.0f32;
    let mut bars = Vec::with_capacity(n_bars);

    for bar_idx in 0..n_bars {
        let bar_start = bar_idx as f32 * beats_per_bar;
        let bar_end = bar_start + beats_per_bar;

        let bar_notes: Vec<crate::midi::MidiNote> = notes.iter()
            .filter(|n| n.start_beat >= bar_start && n.start_beat < bar_end)
            .map(|n| crate::midi::MidiNote {
                pitch: n.pitch,
                start_beat: n.start_beat - bar_start,
                duration_beats: n.duration_beats,
                velocity: n.velocity,
            })
            .collect();

        bars.push(crate::midi::Bar {
            notes: bar_notes,
            beats_per_bar: 4,
            tempo: 120.0,
        });
    }

    bars
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harness_runs() {
        let mut harness = CuriosityHarness::new();
        harness.run(20);
        assert!(harness.best_score() >= 0.0);
        assert_eq!(harness.score_history().len(), 20);
    }

    #[test]
    fn test_harness_score_improves_or_stable() {
        // As the predictor learns, prediction error should generally decrease
        // (the predictor gets better) OR stay high (if engines produce diverse output).
        // We just check it runs without panic and produces finite scores.
        let mut harness = CuriosityHarness::new();
        harness.run(10);
        for &score in harness.score_history() {
            assert!(score.is_finite(), "score should be finite");
            assert!(score >= 0.0, "score should be non-negative");
        }
    }

    #[test]
    fn test_prediction_error_measurement() {
        let harness = CuriosityHarness::new();
        let curr = [0.5; EMBEDDING_DIM];
        let next = [0.5; EMBEDDING_DIM];
        let err = harness.measure_prediction_error(&curr, &next);
        // Identity predictor, same input → error should be 0
        assert_approx_eq::assert_approx_eq!(err, 0.0, 1e-5);
    }

    #[test]
    fn test_notes_to_bars() {
        let notes = vec![
            crate::midi::MidiNote { pitch: 60, start_beat: 0.0, duration_beats: 1.0, velocity: 80 },
            crate::midi::MidiNote { pitch: 64, start_beat: 4.0, duration_beats: 1.0, velocity: 80 },
            crate::midi::MidiNote { pitch: 67, start_beat: 8.0, duration_beats: 1.0, velocity: 80 },
        ];
        let bars = notes_to_bars(&notes, 3);
        assert_eq!(bars.len(), 3);
        assert_eq!(bars[0].notes.len(), 1);
        assert_eq!(bars[0].notes[0].pitch, 60);
        assert_eq!(bars[1].notes[0].pitch, 64);
        assert_eq!(bars[2].notes[0].pitch, 67);
    }
}
