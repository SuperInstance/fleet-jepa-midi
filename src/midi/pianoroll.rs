// src/midi/pianoroll.rs — Piano-roll representation for V-JEPA
//
// A piano roll is a 2D binary matrix: pitch × time.
// This is the representation used by V-JEPA-style approaches
// (treating piano rolls as "video frames").

use crate::midi::Bar;

/// A piano roll for a single bar.
///
/// Dimensions: [128 pitches × N_time_steps]
/// Each cell is the velocity (0 = no note, 1-127 = note active).
#[derive(Debug, Clone)]
pub struct PianoRoll {
    /// Number of time steps per bar (typically 16 or 32).
    pub n_steps: usize,
    /// Velocity values, row-major: [pitch * n_steps + step].
    data: Vec<u8>,
}

impl PianoRoll {
    pub fn new(n_steps: usize) -> Self {
        Self {
            n_steps,
            data: vec![0u8; 128 * n_steps],
        }
    }

    /// Convert a Bar to a PianoRoll with the given number of time steps.
    pub fn from_bar(bar: &Bar, n_steps: usize) -> Self {
        let mut roll = Self::new(n_steps);
        let beats_per_bar = bar.beats_per_bar as f32;
        let step_duration = beats_per_bar / n_steps as f32;

        for note in &bar.notes {
            let start_step = (note.start_beat / step_duration).round() as usize;
            let end_step = ((note.start_beat + note.duration_beats) / step_duration).round() as usize;
            let start_step = start_step.min(n_steps);
            let end_step = end_step.min(n_steps);
            for step in start_step..end_step {
                if step < n_steps {
                    roll.data[note.pitch as usize * n_steps + step] = note.velocity.max(
                        roll.data[note.pitch as usize * n_steps + step]
                    );
                }
            }
        }

        roll
    }

    /// Get velocity at a given pitch and step.
    pub fn get(&self, pitch: usize, step: usize) -> u8 {
        if pitch < 128 && step < self.n_steps {
            self.data[pitch * self.n_steps + step]
        } else {
            0
        }
    }

    /// Set velocity at a given pitch and step.
    pub fn set(&mut self, pitch: usize, step: usize, velocity: u8) {
        if pitch < 128 && step < self.n_steps {
            self.data[pitch * self.n_steps + step] = velocity;
        }
    }

    /// Count active cells (non-zero velocity).
    pub fn active_count(&self) -> usize {
        self.data.iter().filter(|&&v| v > 0).count()
    }

    /// Compute the "density" of the piano roll (fraction of active cells).
    pub fn density(&self) -> f32 {
        self.active_count() as f32 / self.data.len() as f32
    }

    /// Render a text-based view for debugging (low octaves only).
    pub fn debug_view(&self, min_pitch: usize, max_pitch: usize) -> String {
        let mut out = String::new();
        for pitch in (min_pitch..max_pitch).rev() {
            let p_str = format!("{:3} |", pitch);
            out.push_str(&p_str);
            for step in 0..self.n_steps {
                let v = self.get(pitch, step);
                if v > 0 {
                    out.push('#');
                } else {
                    out.push('.');
                }
            }
            out.push('\n');
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::midi::MidiNote;

    #[test]
    fn test_empty_piano_roll() {
        let roll = PianoRoll::new(16);
        assert_eq!(roll.active_count(), 0);
        assert_approx_eq::assert_approx_eq!(roll.density(), 0.0, 1e-6);
    }

    #[test]
    fn test_from_bar() {
        let bar = Bar {
            notes: vec![
                MidiNote { pitch: 60, start_beat: 0.0, duration_beats: 1.0, velocity: 100 },
                MidiNote { pitch: 64, start_beat: 1.0, duration_beats: 1.0, velocity: 80 },
            ],
            beats_per_bar: 4,
            tempo: 120.0,
        };
        let roll = PianoRoll::from_bar(&bar, 16);
        // Step 0-3 should have C4 active, step 4-7 should have E4 active
        assert!(roll.get(60, 0) > 0);
        assert!(roll.get(60, 3) > 0);
        assert!(roll.get(64, 4) > 0);
        assert!(roll.get(64, 7) > 0);
        assert_eq!(roll.get(60, 4), 0); // C4 off by step 4
    }

    #[test]
    fn test_density() {
        let mut roll = PianoRoll::new(4);
        roll.set(60, 0, 100);
        roll.set(60, 1, 100);
        // 2 active out of 128*4 = 512
        let d = roll.density();
        assert_approx_eq::assert_approx_eq!(d, 2.0 / 512.0, 1e-6);
    }

    #[test]
    fn test_debug_view() {
        let bar = Bar {
            notes: vec![
                MidiNote { pitch: 60, start_beat: 0.0, duration_beats: 0.5, velocity: 100 },
            ],
            beats_per_bar: 4,
            tempo: 120.0,
        };
        let roll = PianoRoll::from_bar(&bar, 16);
        let view = roll.debug_view(58, 62);
        assert!(view.contains("60 |"));
        assert!(view.contains('#'));
    }
}
