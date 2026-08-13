// src/jepa/embedding.rs — MIDI to embedding extraction
//
// Initially: 16 hand-crafted features per bar.
// Later: neural encoder (Conformer transformer, 384-dim).

use crate::midi::Bar;

/// The embedding dimension for the initial feature-based encoder.
pub const EMBEDDING_DIM: usize = 16;

/// A 16-dimensional bar embedding.
pub type Embedding = [f32; EMBEDDING_DIM];

/// Named features extracted from a single bar of MIDI.
///
/// These 16 features capture the essential musical qualities that the
/// JEPA layer perceives: energy, tension, density, register, feel, and motion.
/// Each is normalized to roughly [0, 1] unless otherwise noted.
#[derive(Debug, Clone, PartialEq)]
pub struct BarFeatures {
    /// Average number of note onsets per beat slice.
    pub note_density: f32,
    /// Mean MIDI pitch (0-127, normalized to 0-1).
    pub avg_pitch: f32,
    /// Standard deviation of inter-onset intervals (higher = more complex).
    pub rhythmic_complexity: f32,
    /// Harmonic tension estimate via pitch-class dispersion.
    pub harmonic_tension: f32,
    /// Spread of active pitches (max - min).
    pub register_spread: f32,
    /// Mean MIDI velocity (0-127, normalized).
    pub velocity_mean: f32,
    /// Std dev of MIDI velocity.
    pub velocity_std: f32,
    /// Fraction of beats that are syncopated (off-beat).
    pub syncopation: f32,
    /// Melodic contour direction: +1 ascending, 0 stable, -1 descending.
    pub contour_direction: f32,
    /// Mean absolute interval between consecutive notes.
    pub interval_size: f32,
    /// Fraction of the bar with no note activity (rests).
    pub rest_ratio: f32,
    /// Average number of simultaneous notes (chord density).
    pub chord_density: f32,
    /// Activity in the bass register (MIDI 0-47).
    pub bass_register: f32,
    /// Activity in the treble register (MIDI 84-127).
    pub treble_activity: f32,
    /// Dynamic range: max velocity - min velocity (normalized).
    pub dynamic_range: f32,
    /// Ratio of notes with duration > 1 beat (sustained notes).
    pub sustain_ratio: f32,
}

impl BarFeatures {
    /// Convert to a flat embedding array.
    pub fn to_array(&self) -> Embedding {
        [
            self.note_density,
            self.avg_pitch,
            self.rhythmic_complexity,
            self.harmonic_tension,
            self.register_spread,
            self.velocity_mean,
            self.velocity_std,
            self.syncopation,
            self.contour_direction,
            self.interval_size,
            self.rest_ratio,
            self.chord_density,
            self.bass_register,
            self.treble_activity,
            self.dynamic_range,
            self.sustain_ratio,
        ]
    }

    /// Construct from a flat slice (for testing / deserialization).
    pub fn from_slice(s: &[f32]) -> Self {
        assert_eq!(s.len(), EMBEDDING_DIM, "embedding must be {EMBEDDING_DIM}-dim");
        Self {
            note_density: s[0],
            avg_pitch: s[1],
            rhythmic_complexity: s[2],
            harmonic_tension: s[3],
            register_spread: s[4],
            velocity_mean: s[5],
            velocity_std: s[6],
            syncopation: s[7],
            contour_direction: s[8],
            interval_size: s[9],
            rest_ratio: s[10],
            chord_density: s[11],
            bass_register: s[12],
            treble_activity: s[13],
            dynamic_range: s[14],
            sustain_ratio: s[15],
        }
    }
}

/// The JEPA encoder. In v1 this is a deterministic feature extractor.
/// In v2+ it will be a frozen Conformer transformer (384-dim).
pub struct JepaEncoder {
    /// Smoothing factor for exponential moving average (0-1).
    /// α=0.12 → ~880ms time constant at 125ms pulse rate.
    pub smoothing_alpha: f32,
    /// Previous smoothed embedding (for temporal smoothing).
    prev_embedding: Option<Embedding>,
}

impl JepaEncoder {
    pub fn new() -> Self {
        Self {
            smoothing_alpha: 0.12,
            prev_embedding: None,
        }
    }

    /// Extract raw features from a single bar.
    pub fn embed_bar(&self, bar: &Bar) -> Embedding {
        extract_features(bar).to_array()
    }

    /// Extract features with exponential temporal smoothing.
    /// Call this every pulse (125ms) for real-time embedding updates.
    pub fn embed_bar_smoothed(&mut self, bar: &Bar) -> Embedding {
        let raw = self.embed_bar(bar);
        match &self.prev_embedding {
            None => {
                self.prev_embedding = Some(raw);
                raw
            }
            Some(prev) => {
                let a = self.smoothing_alpha;
                let smoothed: [f32; EMBEDDING_DIM] = std::array::from_fn(|i| {
                    a * raw[i] + (1.0 - a) * prev[i]
                });
                self.prev_embedding = Some(smoothed);
                smoothed
            }
        }
    }

    /// Reset the smoothing state (e.g., when starting a new piece).
    pub fn reset(&mut self) {
        self.prev_embedding = None;
    }
}

impl Default for JepaEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract 16 musical features from a bar of MIDI notes.
///
/// All features are normalized to roughly [0, 1] for consistency.
fn extract_features(bar: &Bar) -> BarFeatures {
    let notes = &bar.notes;
    let n_notes = notes.len();

    if n_notes == 0 {
        return BarFeatures {
            note_density: 0.0,
            avg_pitch: 0.5, // neutral center
            rhythmic_complexity: 0.0,
            harmonic_tension: 0.0,
            register_spread: 0.0,
            velocity_mean: 0.0,
            velocity_std: 0.0,
            syncopation: 0.0,
            contour_direction: 0.0,
            interval_size: 0.0,
            rest_ratio: 1.0, // entire bar is rests
            chord_density: 0.0,
            bass_register: 0.0,
            treble_activity: 0.0,
            dynamic_range: 0.0,
            sustain_ratio: 0.0,
        };
    }

    // Note density: notes per slice, normalized by max possible (16 slices)
    let note_density = (n_notes as f32 / 16.0).min(1.0);

    // Average pitch (normalized 0-1)
    let pitches: Vec<u8> = notes.iter().map(|n| n.pitch).collect();
    let avg_pitch_raw: f32 = pitches.iter().map(|&p| p as f32).sum::<f32>() / n_notes as f32;
    let avg_pitch = avg_pitch_raw / 127.0;

    // Rhythmic complexity: std dev of inter-onset intervals
    let mut onsets: Vec<f32> = notes.iter().map(|n| n.start_beat).collect();
    onsets.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let iois: Vec<f32> = onsets.windows(2).map(|w| w[1] - w[0]).collect();
    let rhythmic_complexity = if iois.is_empty() {
        0.0
    } else {
        let mean_ioi: f32 = iois.iter().sum::<f32>() / iois.len() as f32;
        let variance: f32 = iois.iter().map(|x| (x - mean_ioi).powi(2)).sum::<f32>() / iois.len() as f32;
        (variance.sqrt() / 2.0).min(1.0) // normalize: 2 beats is max meaningful std
    };

    // Harmonic tension: pitch-class dispersion
    // Use circular variance on the pitch-class wheel.
    let mut pc_counts = [0u32; 12];
    for &p in &pitches {
        pc_counts[(p % 12) as usize] += 1;
    }
    let total_pc: f32 = pc_counts.iter().map(|&c| c as f32).sum();
    let harmonic_tension = if total_pc > 0.0 {
        // Circular variance: high when pitch classes are spread (more dissonant/tense)
        let mut sum_sin = 0.0;
        let mut sum_cos = 0.0;
        for (i, &count) in pc_counts.iter().enumerate() {
            if count > 0 {
                let angle = 2.0 * std::f32::consts::PI * (i as f32) / 12.0;
                let w = count as f32 / total_pc;
                sum_sin += w * angle.sin();
                sum_cos += w * angle.cos();
            }
        }
        let r = (sum_sin.powi(2) + sum_cos.powi(2)).sqrt();
        1.0 - r // r=1 means concentrated (consonant), r=0 means spread (tense)
    } else {
        0.0
    };

    // Register spread
    let max_pitch = *pitches.iter().max().unwrap() as f32;
    let min_pitch = *pitches.iter().min().unwrap() as f32;
    let register_spread = (max_pitch - min_pitch) / 127.0;

    // Velocity stats
    let velocities: Vec<u8> = notes.iter().map(|n| n.velocity).collect();
    let velocity_mean: f32 = velocities.iter().map(|&v| v as f32).sum::<f32>() / n_notes as f32;
    let velocity_mean_norm = velocity_mean / 127.0;
    let velocity_std = if n_notes > 1 {
        let var: f32 = velocities.iter()
            .map(|&v| (v as f32 - velocity_mean).powi(2))
            .sum::<f32>() / n_notes as f32;
        var.sqrt() / 63.5 // normalize: max meaningful std ~63.5
    } else {
        0.0
    };

    // Syncopation: fraction of notes on off-beats (not on beat 0, 1, 2, 3)
    let syncopated = notes.iter().filter(|n| {
        let beat = n.start_beat.fract();
        beat > 0.01 && (beat - 0.5).abs() > 0.01
    }).count();
    let syncopation = syncopated as f32 / n_notes as f32;

    // Contour direction and interval size
    let mut contour_direction = 0.0f32;
    let mut interval_size = 0.0f32;
    if n_notes >= 2 {
        // Sort by onset for melodic analysis
        let mut sorted: Vec<&crate::midi::MidiNote> = notes.iter().collect();
        sorted.sort_by(|a, b| a.start_beat.partial_cmp(&b.start_beat).unwrap());
        let mut ascending = 0i32;
        let mut descending = 0i32;
        let mut intervals: Vec<i32> = Vec::new();
        for w in sorted.windows(2) {
            let diff = w[1].pitch as i32 - w[0].pitch as i32;
            intervals.push(diff.abs());
            if diff > 0 { ascending += 1; }
            else if diff < 0 { descending += 1; }
        }
        contour_direction = if ascending > descending {
            1.0
        } else if descending > ascending {
            -1.0
        } else {
            0.0
        };
        interval_size = intervals.iter().sum::<i32>() as f32 / intervals.len() as f32 / 24.0; // normalize by 2 octaves
        interval_size = interval_size.min(1.0);
    }

    // Rest ratio: fraction of bar with no note activity
    let mut active_slices = 0u32;
    let n_slices = 16; // 16 sixteenth-note slices per bar (assuming 4/4)
    let beat_per_slice = 4.0 / n_slices as f32;
    for i in 0..n_slices {
        let slice_start = i as f32 * beat_per_slice;
        let slice_end = slice_start + beat_per_slice;
        let any_active = notes.iter().any(|n| {
            n.start_beat < slice_end && (n.start_beat + n.duration_beats) > slice_start
        });
        if any_active {
            active_slices += 1;
        }
    }
    let rest_ratio = 1.0 - (active_slices as f32 / n_slices as f32);

    // Chord density: average simultaneous notes per time slice
    let mut simult_sum = 0u32;
    for i in 0..n_slices {
        let slice_start = i as f32 * beat_per_slice;
        let slice_end = slice_start + beat_per_slice;
        let count = notes.iter().filter(|n| {
            n.start_beat < slice_end && (n.start_beat + n.duration_beats) > slice_start
        }).count();
        simult_sum += count as u32;
    }
    let chord_density = (simult_sum as f32 / n_slices as f32).min(1.0);

    // Bass register activity
    let bass_count = notes.iter().filter(|n| n.pitch < 48).count();
    let bass_register = bass_count as f32 / n_notes as f32;

    // Treble register activity
    let treble_count = notes.iter().filter(|n| n.pitch >= 84).count();
    let treble_activity = treble_count as f32 / n_notes as f32;

    // Dynamic range
    let max_vel = *velocities.iter().max().unwrap() as f32;
    let min_vel = *velocities.iter().min().unwrap() as f32;
    let dynamic_range = (max_vel - min_vel) / 127.0;

    // Sustain ratio: notes with duration > 1 beat
    let sustained = notes.iter().filter(|n| n.duration_beats > 1.0).count();
    let sustain_ratio = sustained as f32 / n_notes as f32;

    BarFeatures {
        note_density,
        avg_pitch,
        rhythmic_complexity,
        harmonic_tension,
        register_spread,
        velocity_mean: velocity_mean_norm,
        velocity_std,
        syncopation,
        contour_direction,
        interval_size,
        rest_ratio,
        chord_density,
        bass_register,
        treble_activity,
        dynamic_range,
        sustain_ratio,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::midi::{Bar, MidiNote};

    fn make_note(pitch: u8, start: f32, dur: f32, vel: u8) -> MidiNote {
        MidiNote { pitch, start_beat: start, duration_beats: dur, velocity: vel }
    }

    #[test]
    fn test_empty_bar() {
        let bar = Bar { notes: vec![], beats_per_bar: 4, tempo: 120.0 };
        let f = extract_features(&bar);
        assert_eq!(f.note_density, 0.0);
        assert_eq!(f.rest_ratio, 1.0);
        assert_eq!(f.velocity_mean, 0.0);
    }

    #[test]
    fn test_single_note() {
        let bar = Bar {
            notes: vec![make_note(60, 0.0, 1.0, 100)],
            beats_per_bar: 4,
            tempo: 120.0,
        };
        let f = extract_features(&bar);
        assert_approx_eq::assert_approx_eq!(f.avg_pitch, 60.0 / 127.0, 0.01);
        assert_approx_eq::assert_approx_eq!(f.velocity_mean, 100.0 / 127.0, 0.01);
        assert!(f.note_density > 0.0);
    }

    #[test]
    fn test_ascending_contour() {
        let bar = Bar {
            notes: vec![
                make_note(60, 0.0, 0.5, 80),
                make_note(64, 0.5, 0.5, 80),
                make_note(67, 1.0, 0.5, 80),
                make_note(72, 1.5, 0.5, 80),
            ],
            beats_per_bar: 4,
            tempo: 120.0,
        };
        let f = extract_features(&bar);
        assert_eq!(f.contour_direction, 1.0); // ascending
        assert!(f.interval_size > 0.0);
    }

    #[test]
    fn test_descending_contour() {
        let bar = Bar {
            notes: vec![
                make_note(72, 0.0, 0.5, 80),
                make_note(67, 0.5, 0.5, 80),
                make_note(64, 1.0, 0.5, 80),
                make_note(60, 1.5, 0.5, 80),
            ],
            beats_per_bar: 4,
            tempo: 120.0,
        };
        let f = extract_features(&bar);
        assert_eq!(f.contour_direction, -1.0); // descending
    }

    #[test]
    fn test_bass_and_treble() {
        let bar = Bar {
            notes: vec![
                make_note(36, 0.0, 2.0, 90),  // bass
                make_note(96, 0.0, 2.0, 90),  // treble
            ],
            beats_per_bar: 4,
            tempo: 120.0,
        };
        let f = extract_features(&bar);
        assert!(f.bass_register > 0.0);
        assert!(f.treble_activity > 0.0);
    }

    #[test]
    fn test_embedding_dimension() {
        assert_eq!(EMBEDDING_DIM, 16);
    }

    #[test]
    fn test_to_array_from_slice_roundtrip() {
        let bar = Bar {
            notes: vec![
                make_note(60, 0.0, 1.0, 100),
                make_note(64, 1.0, 1.0, 80),
            ],
            beats_per_bar: 4,
            tempo: 120.0,
        };
        let f = extract_features(&bar);
        let arr = f.to_array();
        let f2 = BarFeatures::from_slice(&arr);
        assert_eq!(f, f2);
    }

    #[test]
    fn test_smoothing() {
        let bar1 = Bar {
            notes: vec![make_note(60, 0.0, 1.0, 100)],
            beats_per_bar: 4,
            tempo: 120.0,
        };
        let bar2 = Bar {
            notes: vec![make_note(84, 0.0, 1.0, 60)],
            beats_per_bar: 4,
            tempo: 120.0,
        };

        let mut encoder = JepaEncoder::new();
        let emb1 = encoder.embed_bar_smoothed(&bar1);
        let emb2 = encoder.embed_bar_smoothed(&bar2);

        // First embedding should equal the raw embedding
        let raw1 = encoder.embed_bar(&bar1);
        for i in 0..EMBEDDING_DIM {
            assert_approx_eq::assert_approx_eq!(emb1[i], raw1[i], 1e-6);
        }

        // Second embedding should be smoothed (not equal to raw2)
        let raw2 = encoder.embed_bar(&bar2);
        let a = 0.12;
        for i in 0..EMBEDDING_DIM {
            let expected = a * raw2[i] + (1.0 - a) * raw1[i];
            assert_approx_eq::assert_approx_eq!(emb2[i], expected, 1e-5);
        }
    }
}
