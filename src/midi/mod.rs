// src/midi/mod.rs — MIDI I/O

pub mod pianoroll;
pub mod stream;

use midly::{Smf, Timing, TrackEventKind};

/// A single MIDI note with timing in beats.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MidiNote {
    /// MIDI pitch (0-127).
    pub pitch: u8,
    /// Onset time in beats from bar start.
    pub start_beat: f32,
    /// Duration in beats.
    pub duration_beats: f32,
    /// MIDI velocity (0-127).
    pub velocity: u8,
}

/// A bar of music — the atomic unit for JEPA embedding.
#[derive(Debug, Clone)]
pub struct Bar {
    /// All notes active in this bar.
    pub notes: Vec<MidiNote>,
    /// Time signature: beats per bar (4 for 4/4).
    pub beats_per_bar: u32,
    /// Tempo in BPM.
    pub tempo: f32,
}

impl Default for Bar {
    fn default() -> Self {
        Self {
            notes: vec![],
            beats_per_bar: 4,
            tempo: 120.0,
        }
    }
}

/// Convert a parsed SMF (Standard MIDI File) into a vector of bars.
///
/// This is a simple converter: it tracks note-on/note-off events,
/// organizes them into bars based on the PPQ and tempo, and returns
/// bar-by-bar note collections.
pub fn smf_to_bars(smf: &Smf) -> Vec<Bar> {
    let ppq = match smf.header.timing {
        Timing::Metrical(ppq) => ppq.as_int() as f32,
        Timing::FPS(_, _) => 480.0, // fallback
    };
    let beats_per_bar = 4u32; // assume 4/4

    // Collect all note-on/note-off events from track 0 (or first track with notes)
    let mut note_events: Vec<(f32, u8, bool, u8)> = vec![]; // (beat, pitch, is_on, velocity)

    for track in &smf.tracks {
        let mut abs_tick: u32 = 0;
        for event in track {
            abs_tick = abs_tick.saturating_add(event.delta.as_int());
            let beat = abs_tick as f32 / ppq;
            match event.kind {
                TrackEventKind::Midi { message, .. } => {
                    match message {
                        midly::MidiMessage::NoteOn { key, vel } => {
                            if vel > 0 {
                                note_events.push((beat, key.as_int(), true, vel.as_int()));
                            } else {
                                note_events.push((beat, key.as_int(), false, 0));
                            }
                        }
                        midly::MidiMessage::NoteOff { key, .. } => {
                            note_events.push((beat, key.as_int(), false, 0));
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }

    // Sort by beat time
    note_events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Build note pairs: match note-on to note-off
    let mut notes: Vec<MidiNote> = vec![];
    let mut pending: Vec<(f32, u8, u8)> = vec![]; // (start_beat, pitch, velocity)

    for (beat, pitch, is_on, vel) in &note_events {
        if *is_on {
            pending.push((*beat, *pitch, *vel));
        } else {
            // Find matching note-off (most recent same-pitch note-on)
            if let Some(idx) = pending.iter().rposition(|(_, p, _)| *p == *pitch) {
                let (start_beat, _, velocity) = pending.remove(idx);
                let duration = *beat - start_beat;
                notes.push(MidiNote {
                    pitch: *pitch,
                    start_beat,
                    duration_beats: duration.max(0.1),
                    velocity,
                });
            }
        }
    }

    // Handle unterminated notes (note-on with no matching note-off)
    for (start_beat, pitch, velocity) in pending {
        notes.push(MidiNote {
            pitch,
            start_beat,
            duration_beats: 1.0, // default
            velocity,
        });
    }

    // Group into bars
    let bar_duration = beats_per_bar as f32;
    let max_beat = notes.iter()
        .map(|n| n.start_beat + n.duration_beats)
        .fold(0.0f32, f32::max);

    let n_bars = ((max_beat / bar_duration).ceil() as usize).max(1);
    let mut bars = Vec::with_capacity(n_bars);

    for bar_idx in 0..n_bars {
        let bar_start = bar_idx as f32 * bar_duration;
        let bar_end = bar_start + bar_duration;

        let bar_notes: Vec<MidiNote> = notes.iter()
            .filter(|n| n.start_beat >= bar_start && n.start_beat < bar_end)
            .map(|n| MidiNote {
                pitch: n.pitch,
                start_beat: n.start_beat - bar_start,
                duration_beats: (n.start_beat + n.duration_beats).min(bar_end) - n.start_beat,
                velocity: n.velocity,
            })
            .collect();

        bars.push(Bar {
            notes: bar_notes,
            beats_per_bar,
            tempo: 120.0,
        });
    }

    bars
}

#[cfg(test)]
mod tests {
    use super::*;
    use midly::{Header, Format, Timing, Smf, Track, TrackEvent, TrackEventKind, MidiMessage, Ticks};

    #[test]
    fn test_bar_default() {
        let bar = Bar::default();
        assert_eq!(bar.beats_per_bar, 4);
        assert_eq!(bar.tempo, 120.0);
        assert!(bar.notes.is_empty());
    }

    #[test]
    fn test_smf_to_bars() {
        // Build a minimal SMF with 4 quarter notes
        let header = Header::new(Format::SingleTrack, Timing::Metrical(Ticks(480)));
        let mut track = Track::new();

        // Note on C4 at beat 0
        track.push(TrackEvent { delta: Ticks(0), kind: TrackEventKind::Midi {
            channel: midly::num::u4::from(0),
            message: MidiMessage::NoteOn {
                key: midly::num::u7::from(60),
                vel: midly::num::u7::from(100),
            }
        }});
        // Note off C4 at beat 1 (480 ticks)
        track.push(TrackEvent { delta: Ticks(480), kind: TrackEventKind::Midi {
            channel: midly::num::u4::from(0),
            message: MidiMessage::NoteOff {
                key: midly::num::u7::from(60),
                vel: midly::num::u7::from(0),
            }
        }});
        // Note on E4 at beat 1
        track.push(TrackEvent { delta: Ticks(0), kind: TrackEventKind::Midi {
            channel: midly::num::u4::from(0),
            message: MidiMessage::NoteOn {
                key: midly::num::u7::from(64),
                vel: midly::num::u7::from(90),
            }
        }});
        // Note off E4 at beat 2
        track.push(TrackEvent { delta: Ticks(480), kind: TrackEventKind::Midi {
            channel: midly::num::u4::from(0),
            message: MidiMessage::NoteOff {
                key: midly::num::u7::from(64),
                vel: midly::num::u7::from(0),
            }
        }});

        let smf = Smf { header, tracks: vec![track] };
        let bars = smf_to_bars(&smf);

        assert!(!bars.is_empty(), "should have at least one bar");
        assert_eq!(bars[0].notes.len(), 2);
        assert_eq!(bars[0].notes[0].pitch, 60);
        assert_eq!(bars[0].notes[1].pitch, 64);
    }
}
