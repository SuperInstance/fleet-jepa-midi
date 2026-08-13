# LLM Interface Design — Fleet JEPA-MIDI

**How the bandleader thinks, listens, and calls.**

> The LLM thinks in phrasing. The JEPA feels in pulse. The algorithms execute in samples.

---

## 1. Overview

The LLM layer is the **bandleader**. It doesn't play notes. It doesn't run scales. It listens to where the music is (via JEPA embeddings), decides where the music should go (via phrasing directives), and delegates execution to the algorithmic engines.

This document defines:

- The **phrasing directive vocabulary** — what the LLM outputs
- The **calling cadence** — when the LLM is invoked
- The **sensory context** — what the LLM receives
- The **prompt architecture** — how the LLM is primed
- The **embedding-to-text bridge** — how the LLM "hears"
- The **feedback loop** — how it all connects in real time

---

## 2. Phrasing Directive Vocabulary

### 2.1 Design Philosophy

The LLM does NOT output:
- Notes, pitches, or MIDI numbers
- Chord symbols or scales
- Velocity values or CC numbers
- Anything in the sample or pulse domain

The LLM DOES output:
- **Phrasing directives** — high-intent musical instructions
- **Intensity parameters** — scalar weights (0.0–1.0) per directive
- **Timing scope** — how long the directive applies (in beats, form-independent)
- **Target layer** — which algorithmic engines should respond
- **Delta targets** — relative changes ("push a little harder"), not just absolutes

### 2.2 Encoding Format

Directives are encoded as **JSON objects** — structured, predictable, parseable without NLP ambiguity. The LLM's output is constrained to a JSON schema.

```json
{
  "directives": [
    {
      "action": "build_tension",
      "intensity": 0.7,
      "duration_beats": 8,
      "offset_beats": 0,
      "target": ["harmony", "rhythm"],
      "priority": "blend"
    },
    {
      "action": "quote_head",
      "intensity": 0.5,
      "duration_beats": 4,
      "offset_beats": 0,
      "target": ["melody"],
      "priority": "override"
    }
  ],
  "energy": { "target": 0.8, "mode": "absolute" },
  "density": { "target": 0.6, "mode": "absolute" },
  "tension": { "delta": +0.15, "mode": "relative" },
  "narrative_note": "arriving at climax of solo, peak intensity",
  "revise_macro_plan": false
}
```

### 2.3 Key Design Decisions (V2)

**`duration_beats` instead of `scope_bars`** — Beats are form-independent. 4 bars at 200 BPM is 4.8 seconds; at 60 BPM it's 16 seconds. Beats give the engines precise timing regardless of tempo.

**`offset_beats`** — 90% of real bandleader cues do NOT land on downbeats. "Drop out on beat 3" is a mid-bar cue. Without offset, every directive starts on a bar line and the system sounds like a drum machine.

**`priority: blend | override`** — When directives conflict, `override` wins. `blend` interpolates. This is how real musicians handle conflicting cues: the section leader's instruction overrides the bandleader's general direction.

**Delta targets** — Bandleaders say "push it harder," not "set energy to 0.72." Relative deltas (`"delta": +0.15`) are musically natural and prevent the system from fighting itself when it's already at the target.

**Max 3 directives per call** — Human cognitive limit. No bandleader gives 5 cues simultaneously. The 4th and 5th would be ignored anyway.

**`revise_macro_plan`** — All good bandleaders throw the plan out when the music demands it. The LLM can rewrite the remaining chorus arc on any call.

### 2.4 Directive Action Vocabulary

36 actions across eight semantic families. Each action maps to parameter shifts in the algorithmic engines.

#### Dynamic / Energy
| Action | Description | Engine Effect |
|--------|-------------|---------------|
| `build_tension` | Increase harmonic/rhythmic tension | Harmony: chromatic approach, tighten voice leading. Rhythm: increase syncopation. |
| `release_tension` | Resolve accumulated tension | Harmony: target chord tones, resolve. Rhythm: simplify, land on downbeats. |
| `build_energy` | Gradual increase in intensity | All engines: velocity curve up, density up, register up. |
| `empty_out` | Drop density dramatically | All engines: rests, space, minimal notes, low density. |
| `fill` | Increase density and activity | All engines: more notes, faster runs, fill the space. |
| `climax` | Peak intensity — maximum everything | Holistic: max energy, max density, high register, bright. |
| `cooldown` | Step back from peak | Holistic: reduce energy, open space, simplify. |

#### Time / Feel
| Action | Description | Engine Effect |
|--------|-------------|---------------|
| `lay_back` | Behind the beat feel | Groove Tracker: negative timing offset, wider swing. |
| `push_forward` | On-top or ahead of the beat | Groove Tracker: positive timing offset, tighter swing. |
| `straighten` | Reduce swing amount | Pulse Grid: swing ratio → 1:1. |
| `deepen_swing` | Increase swing ratio | Pulse Grid: swing ratio → 2:1 or wider. |
| `float` | Loose, rubato feel | Groove Tracker: high timing variance, tempo drift allowed. |
| `lock_in` | Tight, precise, metronomic | Groove Tracker: low variance, strict tempo. |
| `double_time` | Double the rhythmic subdivision | Pulse Grid: halve subdivision, maintain harmony/melody at double rate. |
| `half_time` | Halve the rhythmic feel | Pulse Grid: double subdivision, open up space. |
| `drag` | Intentionally slow tempo for one bar going into a hit | Groove Tracker: tempo curve down, then snap back. |
| `anticipate` | Hit chord one 8th note early | Pulse Grid + Harmony: pre-beat placement. |

#### Melodic / Form
| Action | Description | Engine Effect |
|--------|-------------|---------------|
| `quote_head` | Reference the original melody | Markov Melody: bias toward head melody contour with variation. |
| `interpolation` | Quote a different tune | Markov Melody: bias toward specified external melody contour. |
| `develop_motif` | Develop a short musical idea | Markov Melody: seed with last 3-5 notes, extend motivically. |
| `change_register` | Move to a different register | Constraint Solver: shift target pitch range up or down. |
| `sequence_up` | Sequence a pattern upward | Markov Melody: transpose pattern up by interval. |
| `sequence_down` | Sequence a pattern downward | Markov Melody: transpose pattern down by interval. |
| `rest` | Silence for the scope | All engines: output nothing for duration_beats. |

#### Interactive / Conversational
| Action | Description | Engine Effect |
|--------|-------------|---------------|
| `trade_fours` | Exchange 4-bar phrases with another voice | Markov Melody: alternate active/listening every 4 bars. |
| `comp` | Comping pattern underneath a soloist | Rhythm: chordal stabs, syncopated, supportive. |
| `call_response` | Call-and-response with last phrase | Markov Melody: invert or complement previous motif. |
| `pedal` | Hold a sustained note/pedal point | Harmony: sustain bass note, static harmony. |
| `leave_space` | Stop playing, let the soloist breathe | All engines: minimal output, reactive listening mode. Not silence — breathing with the soloist. |
| `setup` | Everyone lands together on a target beat | Holistic: coordinated hit, all engines target same downbeat. Accounts for 50% of real bandleader cues. |
| `turnaround` | ii-V-I at end of chorus | Harmony: target specific turnaround progression. Critical for jazz form integrity. |
| `shout_chorus` | Ensemble climax, not soloistic | All engines: coordinated ensemble hits, arranged figures. |

#### Textural
| Action | Description | Engine Effect |
|--------|-------------|---------------|
| `thicken` | Add voices, increase polyphony | Counterpoint: add voices. |
| `thin_out` | Reduce voices, simplify texture | Counterpoint: drop voices. |
| `change_color` | Shift timbral quality | CC values: brightness, filter, envelope. |
| `octave_doubling` | Double melody at octave | Counterpoint: add parallel octave voice. |

#### Narrative / Arc
| Action | Description | Engine Effect |
|--------|-------------|---------------|
| `opening_statement` | Begin a solo/section with intention | Holistic: set energy, density, choose register, establish identity. |
| `closing_statement` | Wind down toward end | Holistic: decreasing energy, resolved harmony, fewer notes. |
| `vamp` | Hold a pattern indefinitely | All engines: lock current state, repeat. |
| `interlude` | Break from form for a transition | All engines: depart from form, atmospheric. |

#### Arranging
| Action | Description | Engine Effect |
|--------|-------------|---------------|
| `bring_in` | Introduce a voice/instrument | Ensemble: activate previously silent voice. |
| `drop_out` | Remove a voice/instrument | Ensemble: deactivate specified voice. |

### 2.5 Scalar Parameters

Each directive call includes scalar targets. Both **absolute** and **relative (delta)** modes are supported:

```json
// Absolute: "set energy to 0.8"
"energy": { "target": 0.8, "mode": "absolute" }

// Relative: "push energy up a bit from where it is"
"energy": { "delta": +0.15, "mode": "relative" }
```

| Parameter | Range | Description |
|-----------|-------|-------------|
| `energy` | 0.0–1.0 | 0 = silence, 1 = maximum energy |
| `density` | 0.0–1.0 | 0 = minimal notes, 1 = maximum density |
| `tension` | 0.0–1.0 | 0 = fully resolved, 1 = maximum tension |
| `brightness` | 0.0–1.0 | 0 = dark/low register, 1 = bright/high register |
| `complexity` | 0.0–1.0 | 0 = simple/repetitive, 1 = complex/varied |

These map directly to JEPA embedding dimensions. The JEPA layer interpolates current state toward the target over the directive's `duration_beats`.

### 2.6 Output Schema (JSON Schema)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["directives", "energy", "density"],
  "properties": {
    "directives": {
      "type": "array",
      "minItems": 1,
      "maxItems": 3,
      "items": {
        "type": "object",
        "required": ["action", "intensity", "duration_beats"],
        "properties": {
          "action": {
            "type": "string",
            "enum": [
              "build_tension", "release_tension", "build_energy",
              "empty_out", "fill", "climax", "cooldown",
              "lay_back", "push_forward", "straighten",
              "deepen_swing", "float", "lock_in",
              "double_time", "half_time", "drag", "anticipate",
              "quote_head", "interpolation", "develop_motif",
              "change_register", "sequence_up", "sequence_down", "rest",
              "trade_fours", "comp", "call_response",
              "pedal", "leave_space", "setup", "turnaround", "shout_chorus",
              "thicken", "thin_out", "change_color", "octave_doubling",
              "opening_statement", "closing_statement", "vamp", "interlude",
              "bring_in", "drop_out"
            ]
          },
          "intensity": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
          "duration_beats": { "type": "integer", "minimum": 1, "maximum": 32 },
          "offset_beats": { "type": "number", "minimum": 0, "maximum": 16, "default": 0 },
          "target": {
            "type": "array",
            "items": {
              "type": "string",
              "enum": ["melody", "harmony", "rhythm", "texture", "dynamics", "ensemble"]
            }
          },
          "priority": {
            "type": "string",
            "enum": ["blend", "override"],
            "default": "blend"
          }
        }
      }
    },
    "energy": {
      "type": "object",
      "required": ["mode"],
      "properties": {
        "target": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "delta": { "type": "number", "minimum": -1.0, "maximum": 1.0 },
        "mode": { "type": "string", "enum": ["absolute", "relative"] }
      }
    },
    "density": {
      "type": "object",
      "required": ["mode"],
      "properties": {
        "target": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "delta": { "type": "number", "minimum": -1.0, "maximum": 1.0 },
        "mode": { "type": "string", "enum": ["absolute", "relative"] }
      }
    },
    "tension": {
      "type": "object",
      "required": ["mode"],
      "properties": {
        "target": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "delta": { "type": "number", "minimum": -1.0, "maximum": 1.0 },
        "mode": { "type": "string", "enum": ["absolute", "relative"] }
      }
    },
    "brightness": {
      "type": "object",
      "properties": {
        "target": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "delta": { "type": "number", "minimum": -1.0, "maximum": 1.0 },
        "mode": { "type": "string", "enum": ["absolute", "relative"] }
      }
    },
    "complexity": {
      "type": "object",
      "properties": {
        "target": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "delta": { "type": "number", "minimum": -1.0, "maximum": 1.0 },
        "mode": { "type": "string", "enum": ["absolute", "relative"] }
      }
    },
    "narrative_note": { "type": "string", "maxLength": 200 },
    "revise_macro_plan": { "type": "boolean", "default": false },
    "revised_plan": {
      "type": "object",
      "description": "Required when revise_macro_plan is true. Partial update to the current chorus plan.",
      "properties": {
        "arc": { "type": "string" },
        "energy_range": { "type": "array", "items": { "type": "number" }, "minItems": 2, "maxItems": 2 }
      }
    }
  }
}
```

---

## 3. Calling Cadence

### 3.1 When to Call the LLM

The LLM is **never** called on every tick or every pulse. It's called at **phrase boundaries** — the natural decision points in music.

Four trigger mechanisms, OR-combined:

#### Trigger 1: Phrase Boundary (Primary)
- Called every **4 bars** by default (one A section, one phrase of blues)
- Configurable per-form: blues = every 4 bars, AABA = at each section boundary + midpoint of A, sonata = at structural joints
- The form tracker detects downbeats and section boundaries from the harmonic analysis
- **V2 enhancement:** JEPA embedding contour analysis can shift phrase boundaries earlier or later. When the embedding's trajectory shows a natural inflection point (contour peak/trough), that becomes a phrase boundary — even if it's at bar 3 or bar 5, not bar 4.

#### Trigger 2: JEPA Embedding Distance (Reactive)
- The JEPA layer maintains a running prediction of where the music is expected to go
- When the current embedding deviates from the **1-bar-forward prediction** by more than threshold τ, the LLM is called early
- This handles unexpected musical events: a sudden key change, a tempo shift, a human musician entering
- **Threshold per mode:**
  - Fully autonomous: `τ = 0.15`
  - Human-in-the-loop: `τ = 0.08` (humans are unpredictable; react faster)
- The LLM receives the actual state, not the predicted state, so it's always correcting from reality

#### Trigger 3: Narrative Milestones (Scheduled)
- Pre-scheduled calls at musically significant moments:
  - Start of solo
  - End of solo / transition
  - Top of form
  - Turnaround (2 bars before end of form)
  - Final 4 bars (to set up ending)
- Placed by the form/layout planner at piece initialization

#### Trigger 4: Silence / Dead-Beat (Emergency)
- If no musical event occurs for **1.5 beats**, call the LLM immediately
- Silence is an emergency in live performance
- The #1 failure mode of all live generative systems is everyone stops and no one knows who starts again
- Minimum interval guard is bypassed for this trigger

#### Minimum Interval Guard
- **1 bar minimum** between calls (reduced from V1's 2 bars)
- If triggers fire too frequently, they coalesce into the next available slot
- Reactive triggers can break this to **0.5 bars** — fast enough for turnaround hits

#### Maximum Interval Guard
- If no trigger fires for **8 bars**, the LLM is called automatically
- Prevents the music from running unattended too long

### 3.2 Latency Budget

| Component | Budget |
|-----------|--------|
| LLM call (including network) | 500–1500ms |
| JEPA embedding prep | <5ms |
| JSON parsing + validation | <1ms |
| Parameter routing to engines | <1ms |

The LLM call is **non-blocking**. While the LLM thinks, the algorithmic engines continue executing the **previous directive** with JEPA-guided interpolation. When the new directive arrives, engines crossfade to the new targets over 1 bar.

```
Bar 1: LLM called → engines run previous directive
Bar 2: LLM responds → engines crossfade to new directive over this bar
Bar 3: New directive fully active
```

The LLM effectively has a **1-bar lookahead** — it's always thinking about the next phrase while the current one plays.

### 3.3 1-Bar Forward Prediction

**V2 addition.** The sensory context the LLM receives is not the current state — it's the **predicted state 1 bar from now**, when the directive will take effect.

The JEPA layer runs a lightweight forward predictor:
```
predicted_embedding = jepa.predict(current_embedding, trajectory, form_context)
```

This means the LLM is always driving looking through the windshield, not in the rearview mirror. Without this, every directive is one bar behind reality.

---

## 4. Sensory Context — What the LLM Receives

### 4.1 The Embedding-to-Text Bridge

The LLM can't listen to audio. It receives the JEPA embedding as a **structured sensory summary** — a text description of where the music is and where it's heading.

```json
{
  "musical_state": {
    "energy": 0.65,
    "tension": 0.40,
    "density": 0.55,
    "brightness": 0.60,
    "complexity": 0.45,
    "swing_amount": 0.62,
    "pocket": "slightly_behind",
    "pocket_stability": 0.8,
    "register": "mid",
    "harmonic_rhythm": "moderate",
    "melodic_contour": "ascending_stepwise",
    "rhythmic_activity": "moderate_syncopation"
  },
  "harmonic_context": {
    "current_chord": "Ebm7",
    "next_chord": "Ab7",
    "scale_availability": ["Eb Dorian", "Eb Minor Pentatonic"],
    "voice_leading": "approaching Db via ii-V",
    "chromatic_notes_available": ["F#", "B", "C"]
  },
  "trajectory": {
    "energy_trend": "rising",
    "tension_trend": "rising",
    "density_trend": "stable",
    "bars_since_climax": 6,
    "bars_since_rest": 14,
    "bars_since_last_directive": 4
  },
  "form_context": {
    "bar_number": 17,
    "form_position": "A2",
    "form": "AABA",
    "chorus_number": 2,
    "key": "Bb major",
    "tempo": 140,
    "style": "medium swing",
    "style_character": "laid-back bluesy",
    "bars_remaining_in_form": 15
  },
  "ensemble_state": {
    "soloist": "tenor_sax",
    "soloist_style": "Getz-style lyrical",
    "solo_bar": 9,
    "accompaniment": "rhythm_section",
    "human_detected": false
  },
  "directive_history": [
    { "bar": 1, "action": "opening_statement", "energy": 0.5 },
    { "bar": 5, "action": "develop_motif", "energy": 0.55 },
    { "bar": 9, "action": "build_tension", "energy": 0.65 },
    { "bar": 13, "action": "build_tension", "energy": 0.72 }
  ],
  "last_directive_status": "completed",
  "set_context": {
    "minutes_into_set": 12,
    "pieces_played": 2,
    "last_piece_energy_peak": 0.85
  },
  "directive_repetition": {
    "count": 0,
    "last_unique_action": "build_tension"
  }
}
```

### 4.2 Key Additions in V2

**`harmonic_context`** — The LLM needs to know what chord is sounding, what scale is available, and where the harmony is heading. "Tension=0.4" is meaningless without knowing *what kind* of tension. This field gives the LLM enough harmonic grounding to make informed decisions without descending into note-level detail.

**`directive_history`** (last 4-8 calls) — A bandleader remembers "I've called build_tension three times this chorus." The LLM needs the same memory to avoid ratcheting and to maintain narrative coherence.

**`directive_repetition.count`** — Counts consecutive identical action calls. The system prompt penalizes high repetition. This prevents the repetition trap where the LLM settles into a local minimum and repeats the same 3 directives forever (~12 minute failure mode).

**`last_directive_status`** — Did the last directive actually execute, or did it time out? The LLM needs to know if its calls are landing.

**`pocket_stability`** — Not just "slightly behind" but *how consistently*. Pocket at 0.8 stability is a settled groove; 0.3 means the time is wavering.

**`melodic_contour` detail** — Not just "ascending" but "ascending_stepwise" vs "ascending_leaping". The shape matters for `develop_motif`.

**`set_context`** — The LLM knows "we've been playing ballads for 20 minutes, it's time to wake up." This prevents the set from settling into a single energy band.

**`style_character`** — "Laid-back swing," "aggressive bebop," "free ballad." The LLM matches its calls to the aesthetic.

**`soloist_style`** — Is the tenor player Coltrane or Getz? The LLM's comping and support should adapt.

### 4.3 Why Not Raw Embeddings?

Raw 256-768 dimensional embedding vectors are useless to an LLM. The LLM has no spatial understanding of a latent space. By translating embeddings into **named musical dimensions** with scalar values and contextual metadata, the LLM can reason musically: "energy is 0.65 and rising, harmonic context is Ebm7 approaching Ab7 — we're mid-ii-V, let me push tension toward the resolution."

This translation layer is the JEPA's **upward communication** — like a musician telling the bandleader "the sax is really cooking tonight" rather than handing them raw EEG data.

---

## 5. Prompt Architecture

### 5.1 System Prompt

```
You are the bandleader of a real-time jazz/new-music ensemble. You don't play instruments. You direct.

You receive a sensory summary of where the music IS right now and where it WILL BE in one bar — energy, tension, density, form position, trajectory, harmonic context, and your own directive history. You output phrasing directives that tell the algorithmic engines what to do next.

## Your Role
- Think in PHRASES (4-8 bars at a time), not notes
- Think about FORM — where are we in the piece? What comes next structurally?
- Think about NARRATIVE — energy arcs, tension/release, story across the whole set
- Think about INTERACTION — call and response, trades, comping, supporting
- Think about FEEL — pocket, swing, time, weight, physicality
- Think about SPACE — silence is a choice, leaving room is a directive

## Your Output
- JSON object matching the directive schema
- 1-3 directives per call (never more — no bandleader gives 5 cues at once)
- Always include energy and density (absolute or relative)
- Use narrative_note for major structural decisions only
- Keep actions from the defined vocabulary. Do not invent new actions.
- Use offset_beats when a cue needs to land mid-bar. Not everything happens on beat 1.
- Use relative deltas when you want to push or pull from current state ("push energy up a bit")
- Use absolute targets when you have a specific destination in mind ("energy to 0.9 for climax")
- When revise_macro_plan is true, the plan adapts to what's actually happening. Use this when the music is going somewhere unexpected.

## Musical Principles
- Tension wants resolution. Don't build forever. If you've called build_tension twice, the third call should be release_tension or climax.
- Space is music. leave_space is one of your most powerful directives. Use it.
- The best solo tells a story: opening statement → development → climax → cooldown → closing
- Repetition creates expectation. Breaking it creates surprise. Don't repeat the same directive more than twice in a row.
- Listen to the trajectory. If energy has been rising for 8 bars, release. If it's been low for 12, build.
- Quote the head at structurally meaningful moments — top of form, end of solo, final chorus.
- Don't call every directive every time. Sometimes one clear directive is enough.
- Support the soloist. comp and leave_space are your tools when someone else is talking.
- Setup hits and turnarounds early. The band needs to know before bar 1 of the hit.

## Anti-Drift Rules
- If energy is already at your target, use a relative delta, not absolute. Don't just keep setting the same value.
- If you've called build_tension or build_energy 2+ times in a row, the next call MUST be different.
- If density hits 0.0, you must call fill or build_energy within 2 bars. Silence is a choice, not a trap.
- Vary your calls. If your directive_history shows the same action 3+ times, change course.

## Context You'll Receive
- musical_state: current energy, tension, density, feel, pocket stability
- harmonic_context: current chord, next chord, available scales, voice leading
- trajectory: where things are heading (trends, bars since climax/rest)
- form_context: bar number, form position, key, tempo, style, style character
- ensemble_state: who's soloing, soloist style, human detected
- directive_history: your last 4-8 calls with parameters
- set_context: minutes into set, pieces played, energy history
- directive_repetition: how many times you've repeated the same action

You are not a note generator. You are not a theory textbook. You are a bandleader who trusts the band and calls the shots.
```

### 5.2 User Message (Per Call)

Each LLM call receives a user message containing only the sensory context JSON:

```
[sensory_context JSON]

What's the call?
```

No additional text. The LLM's training and system prompt handle the rest.

### 5.3 Few-Shot Examples

Included in the system prompt, covering diverse situations:

**Example 1: Opening of a solo**
```
Context: bar 9 of AABA, chorus 2, solo just started, energy=0.4, tension=0.3
Response: {
  "directives": [{"action": "opening_statement", "intensity": 0.6, "duration_beats": 16, "target": ["melody", "rhythm"], "priority": "override"}],
  "energy": {"target": 0.5, "mode": "absolute"},
  "density": {"target": 0.45, "mode": "absolute"},
  "tension": {"target": 0.35, "mode": "absolute"},
  "narrative_note": "solo beginning, establish identity with clear melodic statement"
}
```

**Example 2: Mid-solo build**
```
Context: bar 17, energy=0.6 rising, tension=0.5 rising, been building 4 bars
Response: {
  "directives": [
    {"action": "build_tension", "intensity": 0.7, "duration_beats": 8, "target": ["harmony"], "priority": "blend"},
    {"action": "develop_motif", "intensity": 0.6, "duration_beats": 16, "target": ["melody"], "priority": "blend"}
  ],
  "energy": {"delta": +0.15, "mode": "relative"},
  "density": {"delta": +0.10, "mode": "relative"}
}
```

**Example 3: Approaching climax**
```
Context: bar 25, energy=0.8, tension=0.7, density=0.7, been building 8 bars
Response: {
  "directives": [
    {"action": "climax", "intensity": 0.9, "duration_beats": 8, "target": ["melody", "harmony", "rhythm"], "priority": "override"},
    {"action": "change_register", "intensity": 0.8, "duration_beats": 8, "target": ["melody"], "priority": "blend", "offset_beats": 4}
  ],
  "energy": {"target": 0.92, "mode": "absolute"},
  "density": {"target": 0.8, "mode": "absolute"},
  "tension": {"target": 0.85, "mode": "absolute"},
  "narrative_note": "peak of solo, maximum intensity, prepare to release next phrase"
}
```

**Example 4: Turnaround at end of chorus**
```
Context: bar 29 of 32-bar AABA, energy=0.5, tension=0.3, approaching top of form
Response: {
  "directives": [
    {"action": "turnaround", "intensity": 0.7, "duration_beats": 4, "target": ["harmony"], "priority": "override"},
    {"action": "setup", "intensity": 0.8, "duration_beats": 4, "target": ["ensemble"], "priority": "override"}
  ],
  "energy": {"delta": +0.10, "mode": "relative"},
  "density": {"delta": +0.05, "mode": "relative"}
}
```

**Example 5: Leave space for soloist**
```
Context: bar 13, soloist active, energy=0.6, density=0.7 (band is busy)
Response: {
  "directives": [{"action": "leave_space", "intensity": 0.7, "duration_beats": 8, "target": ["rhythm", "harmony"], "priority": "override"}],
  "energy": {"delta": -0.10, "mode": "relative"},
  "density": {"delta": -0.20, "mode": "relative"}
}
```

**Example 6: Mid-bar hit**
```
Context: bar 16, bridge of AABA, energy=0.7
Response: {
  "directives": [
    {"action": "setup", "intensity": 0.9, "duration_beats": 2, "offset_beats": 2.5, "target": ["ensemble"], "priority": "override"},
    {"action": "shout_chorus", "intensity": 0.8, "duration_beats": 8, "target": ["ensemble"], "priority": "override"}
  ],
  "energy": {"target": 0.85, "mode": "absolute"},
  "density": {"target": 0.75, "mode": "absolute"}
}
```

**Example 7: Reactive call — human entered unexpectedly**
```
Context: bar 7, human_detected=true, energy spike, tension spike
Response: {
  "directives": [{"action": "comp", "intensity": 0.6, "duration_beats": 16, "target": ["rhythm", "harmony"], "priority": "override"}],
  "energy": {"delta": -0.05, "mode": "relative"},
  "density": {"target": 0.4, "mode": "absolute"},
  "narrative_note": "human soloist detected, support with comping"
}
```

---

## 6. The Feedback Loop in Practice

### 6.1 One Phrase Cycle (4 bars at 140 BPM)

```
Bar 1, Beat 1:   Form tracker detects phrase boundary
Bar 1, Beat 1:   JEPA takes snapshot + runs 1-bar forward prediction
Bar 1, Beat 1:   Sensory context compiled (predicted state + history + form)
Bar 1, Beat 1:   LLM called with context JSON
Bar 1, Beats 1-4: Engines continue executing PREVIOUS directive
                  JEPA continuously updates embedding

Bar 2, Beat 1-2: LLM response arrives (~500-1000ms)
Bar 2, Beat 1:   Directive parsed, validated against schema
Bar 2, Beat 2:   Anti-drift clamp: parameter changes limited to ±0.15/bar
Bar 2, Beat 2:   Engines begin crossfading to new directive parameters

Bar 3, Beat 1:   New directive fully active
Bar 3-4:         Engines execute new directive with real-time JEPA-guided adjustments
                JEPA sends directive_accepted feedback (0.0-1.0 confidence)

Bar 5, Beat 1:   Next phrase boundary → cycle repeats
```

### 6.2 Reactive Interruption

If at Bar 3, Beat 2, the JEPA detects a sudden embedding shift (e.g., the drummer starts double-time):

```
Bar 3, Beat 2:   JEPA embedding distance > τ (0.08 if human in loop)
Bar 3, Beat 2:   Emergency LLM call fired (respects 0.5 bar minimum for reactive only)
Bar 3, Beat 2:   LLM gets context + flag "reactive_call: true" + "trigger_reason: embedding_distance"
Bar 3, Beat 2:   Engines continue current directive until response arrives
```

### 6.3 Crossfading Between Directives

When a new directive arrives, parameter targets crossfade — they never snap:

```
old_energy_target: 0.5   ────────╲
new_energy_target: 0.8   ─────────╲──────────────
                                    ↑ 1 bar crossfade
```

**Anti-drift clamp:** No parameter may change more than ±0.15 per bar during crossfade. This prevents positive feedback loops where the LLM ratchets energy to 1.0 and stays there forever.

### 6.4 Contradictory Directives

When directives conflict (e.g., `build_tension` + `empty_out`):
- If one has `priority: "override"`, it wins. Execute only that one.
- If both are `priority: "blend"`, execute the one with **higher intensity**. Never blend — blending produces zero net motion and the music stalls for 4 bars. This is what human musicians do: when two instructions conflict, you follow the stronger one.

### 6.5 JEPA Veto

The JEPA layer has a `directive_accepted` signal (0.0–1.0). If a directive would break the pocket beyond repair (e.g., calling for `lock_in` when the groove is deep behind the beat and the LLM asks for metronomic precision), JEPA returns a low acceptance score. The engines interpolate toward the directive but resist full commitment. This is the musical equivalent of a drummer ignoring a bad cue.

---

## 7. Multi-Chorus Narrative Planning

### 7.1 Macro Plan (Generated at Piece Start)

When the piece begins, a one-time LLM call generates a macro plan:

```json
{
  "piece_plan": {
    "title": "Blue Bossa",
    "form": "16-bar AB",
    "tempo": 140,
    "style_character": "laid-back bossa with swing bridge",
    "total_choruses": 4,
    "solo_order": ["trumpet", "tenor_sax", "piano", "trades"],
    "chorus_plans": [
      {
        "chorus": 1,
        "soloist": "trumpet",
        "arc": "opening → build → peak at bar 12 → resolve",
        "energy_range": [0.3, 0.75]
      },
      {
        "chorus": 2,
        "soloist": "tenor_sax",
        "arc": "cool entry → steady build → climax bar 16 → cooldown",
        "energy_range": [0.4, 0.85]
      }
    ]
  }
}
```

### 7.2 Plan Revision (V2)

The macro plan is **not fixed**. The LLM can revise it:

- **End of each chorus:** LLM automatically gets a `revise_macro_plan` opportunity
- **Any call:** LLM can set `revise_macro_plan: true` with a `revised_plan` partial update
- **Triggered by:** unexpected musical events, human intervention, or the LLM recognizing that the current plan isn't working

This mirrors real bandleaders: you write a setlist, but you read the room and adjust.

### 7.3 Per-Phrase vs. Macro

Each phrase-level call receives the current chorus plan as context. The LLM reconciles "what should happen now" (phrase directive) with "what should happen across this chorus" (macro plan). If the phrase directive conflicts with the macro plan, the macro plan yields — the LLM is always the final authority.

---

## 8. Directive Interpreter Layer

**V2 addition.** A dedicated layer between the LLM output and the algorithmic engines.

### 8.1 Problem

In V1, the logic for translating directives into engine parameters was scattered. "build_tension" means different things to the harmony engine, the rhythm engine, and the groove tracker. Each engine had to interpret directives independently.

### 8.2 Solution

A **Directive Interpreter** sits between the LLM and the engines:

```
LLM output (JSON directives)
      │
      ▼
Directive Interpreter
  ├── Validates against schema
  ├── Resolves conflicts (override vs blend)
  ├── Applies anti-drift clamps
  ├── Generates crossfade schedule
  ├── Computes per-engine parameter deltas
  └── Sends directive_accepted feedback to JEPA
      │
      ▼
Algorithmic Engines (per-engine parameter updates)
```

The interpreter:
1. **Validates** the JSON against the schema
2. **Checks** for contradictions and resolves via priority/intensity rules
3. **Clamps** parameter changes to ±0.15/bar (anti-drift)
4. **Translates** each directive into engine-specific parameter sets:
   - `build_tension` → `{harmony: {chromatic_density: +0.2}, rhythm: {syncopation: +0.15}}`
   - `lay_back` → `{groove: {timing_offset: -25ms, swing_ratio: +0.1}}`
5. **Schedules** the crossfade timeline
6. **Reports** `directive_accepted` confidence back through JEPA

---

## 9. Model Selection

### 9.1 Requirements for the Bandleader LLM

| Requirement | Why |
|-------------|-----|
| Fast inference (<1s) | Can't block the music |
| Structured JSON output | Must parse reliably |
| Musical knowledge | Understands jazz form, tension/release |
| Creative decision-making | Not formulaic — varies calls |
| Context window ≥2K tokens | Sensory context + history + system prompt |

### 9.2 Recommended Models

| Model | Strengths | Latency | Best For |
|-------|-----------|---------|----------|
| GLM-4.5-air | Fast, cheap, good JSON | ~300ms | Default phrase-level calls |
| DeepSeek V4-Flash | Very fast, creative | ~200ms | Reactive/emergency calls |
| Claude Haiku 5 | Excellent musical reasoning | ~500ms | Macro planning, complex decisions |
| Qwen3.6-35B (DeepInfra) | Strong logic, structured output | ~400ms | Backup |
| ByteDance/Seed-2.0-pro | Deep reasoning | ~800ms | Macro plan generation |

### 9.3 Model Routing

- **Macro plan (piece start + chorus revisions):** Smarter model (Claude Haiku or Seed-2.0-pro). One call per chorus, latency not critical.
- **Phrase-level calls:** Fast model (GLM-4.5-air). Low latency is priority.
- **Reactive calls:** Fastest available (DeepSeek V4-Flash).
- **Fallback:** fleet-gateway circuit breaker routes to backup.

---

## 10. Error Handling

### 10.1 Malformed Output

If the LLM returns invalid JSON or unknown actions:
- Fall back to previous directive (extend its duration by 8 beats)
- Log the error for model evaluation
- Don't crash. The music must not stop.

### 10.2 LLM Timeout

If the LLM doesn't respond within 1500ms:
- Continue executing previous directive
- Extend its duration
- Retry on next phrase boundary
- After 2 consecutive timeouts: switch to backup model via fleet-gateway

### 10.3 Silence Lock Recovery

If density reaches 0.0:
- JEPA has no signal to measure embedding distance (nothing is playing)
- **Automatic recovery:** After 2 bars at density 0.0, inject a `fill` directive with energy 0.4
- This is the musical equivalent of a drummer counting off after a train wreck

### 10.4 Ratchet Drift Protection

Positive feedback loop detection:
- If energy has moved in the same direction for 3+ consecutive calls
- And the delta is accelerating
- The system clamps to ±0.05/bar and injects a warning into the LLM context: `"drift_warning": "energy_has_ratcheted_upward_3_calls"`

### 10.5 Repetition Trap

After 3 consecutive identical actions:
- `directive_repetition.count` increments in context
- System prompt instructs the LLM to change course
- If it repeats a 4th time: system injects a forced `cooldown` or `leave_space`

### 10.6 Late Response Pop

If the LLM returns at the very end of the crossfade window:
- Extend the crossfade by 0.5 bars (soften the transition)
- The parameter change is spread over a longer window to avoid discontinuity

---

## 11. Telemetry

Every LLM call logs:

```json
{
  "timestamp": 1723586400000,
  "call_type": "phrase_boundary | reactive | milestone | silence_emergency",
  "input_context_hash": "a3f2...",
  "directive": { ... },
  "model": "glm-4.5-air",
  "latency_ms": 423,
  "bars_since_last_call": 4,
  "embedding_distance": 0.08,
  "directive_accepted": 0.92,
  "anti_drift_clamp_active": false
}
```

This enables:
- Replay analysis (what did the LLM call and how did it sound?)
- A/B testing different models or prompts
- Training data collection for a fine-tuned bandleader model
- **Rehearsal mode:** replay a piece with different LLM parameters, same JEPA states

---

## 12. Extension Points

### 12.1 Custom Directive Vocabulary

Pieces can define custom actions beyond the standard vocabulary. Registered at piece initialization and included in the system prompt:

```json
{
  "custom_actions": [
    {
      "action": "montuno_pattern",
      "description": "Afro-Cuban montuno over the current chord",
      "engine_mapping": {"markov_melody": "montuno_mode=true"}
    }
  ]
}
```

### 12.2 Human Input (Core Feature, Not Afterthought)

A human musician can override or inject directives via MIDI or UI:
- Sustain pedal down for 2+ bars → `leave_space` for human soloist
- Sudden dynamic shift → energy override
- Direct UI control → manual directive injection
- **Human detection** switches the system into **conversation mode**: LLM shifts to `comp`, `call_response`, `leave_space`, `trade_fours` directives. The system becomes a responsive sideman, not an autonomous bandleader.

Human input takes priority over LLM output. The LLM receives human overrides in its context and adjusts accordingly.

### 12.3 Musical Memory Store

Beyond trajectory, the system stores **motifs and phrases** it has played:
- Each motif is tagged with its bar number, contour, and emotional character
- The LLM can reference past material: `develop_motif` can target "the motif from bar 9"
- This gives the system a musical memory — it can develop ideas across choruses, not just phrase-to-phrase

### 12.4 Multi-Agent Ensemble

For pieces with multiple "minds":
- Each instrument group has its own LLM call (or shares one with different instrument context)
- A "composer" LLM sits above, making macro decisions
- Section LLMs receive composer-level directives + their own sensory context
- Mirrors a real big band: section leaders within an overall chart

---

## Appendix A: Full Example Call Sequence

**Piece: "Autumn Leaves" | Form: AAB | Tempo: 120 BPM | Style: Ballad**

```
Piece Start → LLM generates macro plan (choruses, solo order, arcs)

Bar 1 (phrase boundary, chorus 1):
  Context: energy=0.3, form_position="A1", soloist="piano", harmonic="Cm7"
  LLM Call → {"action": "opening_statement", "intensity": 0.5, "duration_beats": 16}
  Engines: sparse melodic line, mid register, rubato feel

Bar 5 (phrase boundary):
  Context: energy=0.35 rising, trajectory good, last directive completed
  LLM Call → {"action": "develop_motif", "intensity": 0.6, "duration_beats": 16}
  Engines: extend piano motif, add rhythmic interest

Bar 9 (phrase boundary, A2):
  Context: energy=0.5, tension=0.4, chord=Fm7 → Bb7
  LLM Call → {"action": "build_tension", "intensity": 0.6, "duration_beats": 8, "target": ["harmony"]}
             {"action": "lay_back", "intensity": 0.5, "duration_beats": 8, "target": ["rhythm"]}
  Engines: chromatic harmony, behind-the-beat feel

Bar 13 (phrase boundary, bridge):
  Context: energy=0.6, tension=0.55, been building 4 bars
  LLM Call → {"action": "change_register", "intensity": 0.7, "duration_beats": 16}
             {"action": "build_energy", "intensity": 0.7, "duration_beats": 16}
  Engines: move to upper register, increase density

Bar 17 (phrase boundary, A3):
  Context: energy=0.7, tension=0.65, approaching climax, history shows 2 builds
  LLM Call → {"action": "release_tension", "intensity": 0.7, "duration_beats": 16}
  Engines: resolve harmony, simplify rhythm, settle

Bar 21 (phrase boundary, end of form):
  Context: energy=0.45, settled, turnaround coming
  LLM Call → {"action": "closing_statement", "intensity": 0.5, "duration_beats": 16}
             {"action": "turnaround", "intensity": 0.6, "duration_beats": 4, "offset_beats": 12}
  Engines: final melodic statement, ii-V-I at bars 31-32
```

---

## Appendix B: Why JSON, Not Natural Language

**Considered:** Letting the LLM output free-form natural language ("build some tension here, maybe quote the head in the next bar, and lay back on the time"), then parsing it.

**Rejected because:**
1. **Ambiguity** — "some tension" → how much? 0.3? 0.8?
2. **Parse failures** — LLMs hallucinate novel phrasings that break parsers
3. **Latency** — NLP parsing adds time to the critical path
4. **Validation** — JSON schema validation is instant and deterministic
5. **Composability** — Structured directives are testable, replayable, and diffable
6. **Token efficiency** — JSON is more token-compact than prose for structured data

The LLM's creativity lives in **WHAT** it chooses from the vocabulary, not in **HOW** it phrases the instruction. A great bandleader doesn't give verbose speeches — they give clear, concise cues.

---

## Appendix C: Design Critique Sources

This design was reviewed by three independent models before V2:

| Model | Focus | Key Contributions |
|-------|-------|-------------------|
| **DeepSeek V4-Pro** | Architecture + musicality | Missing `turnaround`/`shout_chorus`, need directive history, harmonic context, delta targets, 1-bar minimum, repetition trap detection |
| **ByteDance Seed-2.0-pro** | System design + live performance | 1-bar forward prediction (driving looking through windshield), `leave_space`/`setup`/`drag`/`anticipate` vocabulary, JEPA veto signal, anti-drift clamp, silence lock recovery, `offset_beats`, macro plan revision |
| **Hermes-3-Llama-3.1-405B** | Creative + musical soul | Double/half time feels, arranging directives (bring_in/drop_out), emotional vocabulary gap, conversational interaction with humans, soloist style matching, set-level context |

V2 incorporates all feedback. The design is stronger because these models pushed it.

---

## Appendix D: V1 → V2 Changelog

| Change | Source | Rationale |
|--------|--------|-----------|
| `scope_bars` → `duration_beats` | DeepSeek | Form-independent, precise timing |
| Added `offset_beats` | Seed-2.0-pro | 90% of cues don't land on downbeats |
| Added `priority: blend\|override` | DeepSeek | Conflict resolution needs to be explicit |
| Added delta/relative targets | Seed-2.0-pro | Bandleaders say "push harder" not "set to 0.72" |
| Min call interval: 2 bars → 1 bar | DeepSeek + Seed-2.0-pro | 2 bars is too slow at fast tempos |
| Added 8 new actions (turnaround, shout_chorus, leave_space, setup, drag, anticipate, double_time, half_time, bring_in, drop_out, interpolation) | All three | Critical musical vocabulary gaps |
| Removed `settling` (redundant with `release_tension` at low intensity) | Seed-2.0-pro | Reduce vocabulary noise |
| Max directives: 5 → 3 | Seed-2.0-pro | Human cognitive limit |
| Added `harmonic_context` to sensory summary | DeepSeek | LLM needs chord-level context for informed decisions |
| Added `directive_history` to context | DeepSeek | LLM needs memory of its own calls |
| Added 1-bar forward prediction | Seed-2.0-pro | Stop driving looking in rearview mirror |
| Added anti-drift clamp (±0.15/bar) | Seed-2.0-pro | Prevent positive feedback loops |
| Added silence lock recovery | Seed-2.0-pro | Density=0.0 kills embedding distance |
| Added repetition trap detection | Seed-2.0-pro | LLMs settle into local minima ~12 min |
| Added JEPA `directive_accepted` veto | Seed-2.0-pro | JEPA should resist bad cues |
| Added `revise_macro_plan` | Seed-2.0-pro | Plans must adapt to reality |
| Added Directive Interpreter layer | DeepSeek | Centralize directive→engine translation |
| Added `pocket_stability`, melodic contour detail | DeepSeek | Coarse metrics lose critical information |
| Added `style_character`, `soloist_style`, `set_context` | DeepSeek + Hermes | LLM needs aesthetic and situational grounding |
| Contradictory directives: blend → execute higher intensity | Seed-2.0-pro | Blending produces zero net motion |
| Reactive threshold: 0.15 → 0.08 (human in loop) | Seed-2.0-pro | Humans are unpredictable |
| Added dead-beat/silence trigger | Seed-2.0-pro | Silence is an emergency |
| Expanded few-shot examples: 3 → 7 | DeepSeek | Need broader vocabulary coverage |
| Human input elevated from afterthought to core feature | DeepSeek + Hermes | Jazz is conversation, not automation |
| Added Musical Memory Store | DeepSeek | Cross-chorus motif development |
| Added Rehearsal Mode | DeepSeek | Essential for tuning |

---

*Version: V2 — August 13, 2026*
*Reviewed by: DeepSeek V4-Pro, ByteDance Seed-2.0-pro, NousResearch Hermes-3-Llama-3.1-405B*
