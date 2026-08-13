# Fleet JEPA-MIDI

**A three-layer real-time music intelligence system.**

The LLM thinks in phrasing. The JEPA feels in pulse. The algorithms execute in samples.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              LLM (per-phrasing-unit)                 │
│   "Build tension here. Resolve in two bars.          │
│    Quote the bridge melody. Trade fours."            │
│   Called on the pulse, not on every tick.            │
└──────────────────────┬──────────────────────────────┘
                       │ direction / intent / phrasing
                       ▼
┌─────────────────────────────────────────────────────┐
│              JEPA-MIDI (high sample rate)            │
│   Latent embedding of where the music IS right now.  │
│   Energy, direction, pocket, tension, swing.         │
│   Feeds sensory read UP to LLM.                      │
│   Feeds parameter adjustments DOWN to algorithms.    │
└──────────────────────┬──────────────────────────────┘
                       │ parameters / targets / constraints
                       ▼
┌─────────────────────────────────────────────────────┐
│         Algorithmic Engines (sub-millisecond)         │
│   Pulse grid, counterpoint, groove tracker,           │
│   Markov chains, constraint solvers.                  │
│   Pure math. No thinking. Just execution.             │
└─────────────────────────────────────────────────────┘
```

## The Three Timescales

| Layer | Timescale | What it does |
|-------|-----------|-------------|
| LLM | per phrase (1-4 bars) | Thinks: form, direction, dynamics, quotes |
| JEPA | per pulse (16th notes ~125ms) | Feels: pocket, energy, tension, swing |
| Algorithms | per sample (<1ms) | Executes: notes, velocities, CC values |

## Training Data

- **Lakh MIDI Dataset** (176,000+ files)
- **MAESTRO** (200 hours virtuosic piano)
- **Hooktheory Corpus** (melody + harmony pairs)
- **SuperInstance Fakebook** (our own TapScript transcriptions)

## Layers in Detail

### Layer 1: JEPA-MIDI (Perception)

A Joint Embedding Predictive Architecture that learns the latent space of music from MIDI. Not a generator — a perceiver. It answers: *what does the music feel like right now?*

- Input: MIDI stream at high sample rate (ppq=960 or higher)
- Output: embedding vector (256-768 dims) updated per pulse
- Training: self-supervised on MIDI corpus, predicting next-bar embedding from current-bar + context
- The embedding encodes: energy level, harmonic tension, rhythmic tightness, melodic direction, swing amount, register, density

### Layer 2: Algorithmic Engines (Execution)

Existing and new MIDI generation algorithms that run fast:

- **Pulse Grid** — rhythmic subdivision and swing
- **Counterpoint Analyzer** — voice leading rules
- **Groove Tracker** — timing/velocity humanization
- **Markov Melody** — style-aware melody generation
- **Constraint Solver** — target notes, avoid notes, ranges
- **Tension Model** — harmonic tension over time

Each engine takes parameters from the JEPA+LLM layer and outputs MIDI events in real-time.

### Layer 3: LLM (Direction)

Called when the phrasing requires a decision — not on every note:

- Input: JEPA embedding (where the music is) + musical context (key, form, bar number) + prompt
- Output: phrasing directive ("build tension", "quote the head", "lay back on the time", "empty out for two bars then fill")
- Called every 1-4 bars, NOT every tick
- The LLM is the bandleader. It doesn't play. It calls.

## The Feedback Loop

```
JEPA reads current state ──► LLM decides direction ──► Algorithms execute
        ▲                                                        │
        └──────────── JEPA reads new state ◄────────────────────┘
```

The loop runs at the pulse rate. The LLM runs at the phrasing rate. The algorithms run at the sample rate. Three clocks, one instrument.

## Relation to Existing Fleet

- **fleet-gateway**: routes LLM calls with circuit breaker
- **fleet-memory**: stores MIDI corpus embeddings for JEPA training
- **TapScript**: the notation system this instrument speaks
- **CNS escalation pattern**: same three-tier architecture (reflex → trained → reasoned)

## Two Complementary Flows

This repo and [fleet-ensemble](https://github.com/SuperInstance/fleet-ensemble) are companion systems:

1. **JEPA-MIDI (this repo)** — **Construction**: sound → JEPA perceives feel → LLM thinks in phrasing → algorithms execute → MIDI emerges. Building music FROM feel.

2. **Fleet Ensemble** — **Performance**: MIDI score → performer agents render it with intelligence → JEPA director shapes the feel → output is more than notes, it's a *performance*. Rendering MIDI AS more than notes on a page.

Both are modular and agnostic. The JEPA, the performer, the rendering system — all pluggable. The magic is in the synergy: a higher-level intelligence deciding the *manner* of playing.

## Status

**Concept phase.** Repo created Aug 13, 2026. Design in progress.

Also see the [infrastructure redesign](https://github.com/SuperInstance/AI-Writings/tree/main/infrastructure) for how this fits the broader fleet architecture.

## License

MIT
