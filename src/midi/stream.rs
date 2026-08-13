// src/midi/stream.rs — Real-time MIDI stream
//
// A circular buffer for live MIDI events, used by the JEPA inference path
// to maintain the last ~2 seconds of musical context.

use crate::midi::MidiNote;
use std::collections::VecDeque;

/// A circular buffer for live MIDI input.
///
/// Holds note events with timestamps, pruning old events based on
/// a configurable window duration. Designed for the JEPA real-time
/// embedding engine which needs ~2048ms of context.
pub struct MidiStream {
    /// Duration of the buffer in milliseconds.
    pub window_ms: f32,
    /// Incoming events: (timestamp_ms, note).
    events: VecDeque<(f32, MidiNote)>,
}

impl MidiStream {
    pub fn new(window_ms: f32) -> Self {
        Self {
            window_ms,
            events: VecDeque::new(),
        }
    }

    /// Add a note event with a timestamp (in milliseconds from stream start).
    pub fn ingest(&mut self, timestamp_ms: f32, note: MidiNote) {
        self.events.push_back((timestamp_ms, note));
        self.prune(timestamp_ms);
    }

    /// Remove events older than window_ms from the latest timestamp.
    fn prune(&mut self, latest_ms: f32) {
        let cutoff = latest_ms - self.window_ms;
        while let Some(&(ts, _)) = self.events.front() {
            if ts < cutoff {
                self.events.pop_front();
            } else {
                break;
            }
        }
    }

    /// Get all events in the current window.
    pub fn events(&self) -> impl Iterator<Item = &(f32, MidiNote)> {
        self.events.iter()
    }

    /// Number of events currently buffered.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Convert the buffered events to a Bar (for JEPA embedding).
    ///
    /// Maps timestamps to beat positions assuming a given tempo.
    pub fn to_bar(&self, tempo: f32) -> crate::midi::Bar {
        let beats_per_second = tempo / 60.0;
        let ms_per_beat = 1000.0 / beats_per_second;

        let notes: Vec<MidiNote> = self.events.iter()
            .map(|(ts, n)| {
                let beat_offset = ts / ms_per_beat;
                MidiNote {
                    pitch: n.pitch,
                    start_beat: beat_offset % 4.0, // fold into bar
                    duration_beats: n.duration_beats,
                    velocity: n.velocity,
                }
            })
            .collect();

        crate::midi::Bar {
            notes,
            beats_per_bar: 4,
            tempo,
        }
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.events.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(pitch: u8, vel: u8) -> MidiNote {
        MidiNote { pitch, start_beat: 0.0, duration_beats: 0.5, velocity: vel }
    }

    #[test]
    fn test_pruning() {
        let mut stream = MidiStream::new(1000.0); // 1 second window
        stream.ingest(0.0, note(60, 100));
        stream.ingest(500.0, note(62, 90));
        stream.ingest(1100.0, note(64, 80));

        // Event at t=0 should be pruned (older than 100ms cutoff = 1100-1000=100)
        assert_eq!(stream.len(), 2); // only 500ms and 1100ms events
    }

    #[test]
    fn test_empty_stream() {
        let stream = MidiStream::new(2000.0);
        assert!(stream.is_empty());
        assert_eq!(stream.len(), 0);
    }

    #[test]
    fn test_to_bar() {
        let mut stream = MidiStream::new(2000.0);
        stream.ingest(0.0, note(60, 100));
        stream.ingest(500.0, note(64, 90));

        let bar = stream.to_bar(120.0);
        assert_eq!(bar.notes.len(), 2);
        assert_eq!(bar.tempo, 120.0);
    }

    #[test]
    fn test_clear() {
        let mut stream = MidiStream::new(2000.0);
        stream.ingest(0.0, note(60, 100));
        assert!(!stream.is_empty());
        stream.clear();
        assert!(stream.is_empty());
    }
}
