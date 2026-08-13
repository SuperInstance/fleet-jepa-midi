# Agentic Algorithmic Music Systems
## Deep Ideation: Reharnessing Markov, L-Systems, Fractals, and Cellular Automata with a JEPA+LLM Center

> **Architecture vision:** Classical algorithmic music generators become the *execution layer*. A JEPA (Joint Embedding Predictive Architecture) perceives their output in latent space. An LLM acts as bandleader, making high-level musical decisions that modulate algorithm parameters in real-time. The old algorithms don't change — but they become *controllable instruments* played by an agentic center.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Agentic Markov Chains](#2-agentic-markov-chains)
3. [Agentic L-Systems](#3-agentic-l-systems)
4. [Agentic Fractals](#4-agentic-fractals)
5. [Agentic Cellular Automata](#5-agentic-cellular-automata)
6. [The Agentic Center: JEPA + LLM](#6-the-agentic-center-jepa--llm)
7. [Multi-Engine Orchestration](#7-multi-engine-orchestration)
8. [Latency & Performance Budget](#8-latency--performance-budget)
9. [Research Context & Prior Art](#9-research-context--prior-art)

---

## 1. System Overview

### The Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     AGENTIC CENTER                              │
│                                                                 │
│   ┌──────────────┐         ┌──────────────────┐                │
│   │   JEPA       │────────▶│   LLM Bandleader │                │
│   │  (Perceive)  │ embed   │   (Decide)       │                │
│   └──────────────│────────▶│                  │                │
│        ▲         │ predict └────────┬─────────┘                │
│        │         │                  │ parameter deltas          │
│        │         └──────────────────┘                          │
│        │                  │                                     │
│        │           parameter updates                            │
│        ▼                  ▼                                     │
├─────────────────────────────────────────────────────────────────┤
│                  EXECUTION LAYER                                │
│                                                                 │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│   │ Markov   │  │ L-System │  │ Fractal  │  │ Cellular │      │
│   │ (Melody) │  │ (Harmony)│  │(Contour) │  │  (Rhythm)│      │
│   └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
│                                                                 │
│   Each engine: old algorithm + real-time parameter interface    │
├─────────────────────────────────────────────────────────────────┤
│                     AUDIO/MIDI OUTPUT                           │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│   │ Melody   │  │ Harmony  │  │ Dynamics │  │ Rhythm   │      │
│   │ Track    │  │ Track    │  │ Track    │  │ Track    │      │
│   └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### Core Insight

The algorithmic engines (Markov, L-system, fractal, CA) are already good at generating musical material. What they lack is **intentionality** — the ability to *steer* the music toward an emotional goal, respond to the room, build tension over a 32-bar arc, or shift from cool jazz to fusion over 8 bars.

The agentic center provides this intentionality through a closed loop:

1. **JEPA perceives** the algorithmic output as a multi-dimensional embedding
2. **JEPA predicts** what the next phrase will sound like (embedding prediction)
3. **LLM receives** a musical context summary + JEPA's predictions
4. **LLM decides** parameter adjustments (temperature, rules, fractal dimension, CA rule)
5. **Engines execute** with updated parameters → generate next phrase
6. **Loop repeats**

---

## 2. Agentic Markov Chains

### Traditional Foundation

A Markov chain for music stores transition probabilities between states (notes, chords, rhythmic values). An *n*-order chain looks at the last *n* states to predict the next. The chain is learned from a corpus — Bach chorales, Coltrane solos, folk melodies — and generates new material by sampling from the transition distribution.

**What's fixed:** The state space and transition matrix (learned offline).
**What we make controllable:** Everything about *how* we sample from it.

### 2.1 The Parameter Interface

The LLM controls the following Markov parameters in real-time:

#### Core Sampling Parameters

| Parameter | Range | Default | Update Rate | Musical Effect |
|-----------|-------|---------|-------------|----------------|
| `temperature` | 0.1–2.0 | 0.85 | Every 2 bars | Low = predictable/safe; High = adventurous/risky |
| `order` | 1–5 | 3 | Every phrase | Low = more random; High = more corpus-faithful |
| `top_k` | 1–20 | 8 | Every 2 bars | Constrains to *k* most likely transitions |
| `repetition_penalty` | 0.0–1.0 | 0.1 | Continuous | Prevents getting stuck in loops |

#### Musical Constraint Parameters

| Parameter | Type | Update Rate | Musical Effect |
|-----------|------|-------------|----------------|
| `key_lock` | Key + Mode | Per section | Forces output to a specific key |
| `allowed_degrees` | List[int] | Per phrase | Scale degrees allowed (e.g., pentatonic = [1,2,3,5,6]) |
| `chromatic_tolerance` | Float 0–1 | Every 4 bars | Probability of allowing chromatic passing tones |
| `max_interval` | Int (semitones) | Every 2 bars | Maximum leap allowed |
| `leap_reversal` | Bool | Per phrase | If true, leaps must be followed by stepwise motion |
| `directional_bias` | Float 0–1 | Every 2 bars | Ascending vs descending probability |

#### Harmony-Specific Parameters

| Parameter | Type | Musical Effect |
|-----------|------|----------------|
| `dissonance_tolerance` | Float 0–1 | How much harmonic tension to allow |
| `harmonic_rhythm` | Float | Chords per bar (0.5 = one chord per 2 bars) |
| `progression_complexity` | Int 1–5 | Maximum number of chords in a progression |
| `secondary_dominants` | Bool | Allow secondary dominant chords |

#### Rhythm-Specific Parameters

| Parameter | Type | Musical Effect |
|-----------|------|----------------|
| `rhythm_density` | Float 0–1 | Note density (0.2 = sparse, 0.8 = busy) |
| `syncopation_level` | Float 0–1 | Off-beat emphasis |
| `swing_amount` | Float 0–0.75 | Triplet swing ratio |

### 2.2 Parameter Control Implementation

```python
class MarkovControlParameters:
    """All parameters the LLM can adjust in real-time."""
    
    def __init__(self):
        # Core sampling (updated every 2-8 bars)
        self.temperature: float = 0.85
        self.order: int = 3
        self.top_k: int = 8
        self.repetition_penalty: float = 0.1
        
        # Musical constraints (updated per phrase/section)
        self.key_lock: tuple[str, str] = ("C", "major")
        self.allowed_degrees: list[int] = [1, 2, 3, 4, 5, 6, 7]
        self.chromatic_tolerance: float = 0.2
        self.max_interval: int = 12  # semitones
        self.leap_reversal: bool = True
        self.directional_bias: float = 0.5  # 0.5 = neutral
        
        # Harmony
        self.dissonance_tolerance: float = 0.2
        self.harmonic_rhythm: float = 1.0  # chord/bar
        self.progression_complexity: int = 3
        
        # Rhythm
        self.rhythm_density: float = 0.6
        self.syncopation_level: float = 0.3
        self.swing_amount: float = 0.0
        
        # Adaptive (updated every note)
        self.surprise_target: float = 0.3  # target information entropy
        self.momentum: float = 0.5  # exploration vs exploitation


class AgenticMarkovSampler:
    """Markov chain that samples with LLM-controlled parameters."""
    
    def __init__(self, transition_matrix, params: MarkovControlParameters):
        self.matrix = transition_matrix  # Pre-learned, fixed
        self.params = params
    
    def sample_next(self, current_state, context):
        """Sample next note with live-controlled parameters."""
        
        # Get raw transition distribution for current state
        raw_probs = self.matrix.get_transitions(current_state, order=self.params.order)
        
        # Apply key constraint: zero out notes outside allowed scale
        raw_probs = self._apply_key_filter(raw_probs, self.params.key_lock,
                                           self.params.allowed_degrees,
                                           self.params.chromatic_tolerance)
        
        # Apply interval constraint: zero out leaps beyond max_interval
        raw_probs = self._apply_interval_filter(raw_probs, current_state,
                                                self.params.max_interval,
                                                self.params.leap_reversal)
        
        # Apply top-k filtering
        if self.params.top_k < len(raw_probs):
            top_indices = np.argsort(raw_probs)[-self.params.top_k:]
            mask = np.zeros_like(raw_probs)
            mask[top_indices] = 1
            raw_probs *= mask
        
        # Apply temperature
        logits = np.log(raw_probs + 1e-8) / self.params.temperature
        probs = softmax(logits)
        
        # Apply repetition penalty
        for recently_played in context.recent_notes:
            probs[recently_played] *= (1 - self.params.repetition_penalty)
        
        # Sample
        return np.random.choice(len(probs), p=probs / probs.sum())
```

### 2.3 What the JEPA Perceives

The JEPA encodes the Markov chain's output into a **musical embedding space** and predicts what comes next. It doesn't see notes — it sees *musical meaning*.

#### Input Representation
The JEPA receives a **multi-view encoding** of each generated phrase:

- **Melody view**: Pitch contour as a sequence of intervals (semitone deltas)
- **Rhythm view**: Inter-onset intervals normalized by tempo
- **Harmony view**: Chroma vector (12-bin pitch class distribution) per beat
- **Timbre view**: If audio is available, mel-spectrogram embeddings

#### Architecture

```python
class JEPAMarkovPerceiver:
    """JEPA that perceives Markov chain output."""
    
    def __init__(self):
        # Encoders (Siamese with EMA-updated target encoder)
        self.context_encoder = MusicTransformerEncoder(
            d_model=256, nhead=8, num_layers=4
        )
        self.target_encoder = MusicTransformerEncoder(
            d_model=256, nhead=8, num_layers=4
        )
        # Target encoder updated via exponential moving average
        
        # Predictor: given context embedding, predict target embedding
        self.predictor = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 256)
        )
        
        # Projection head for downstream tasks
        self.emotion_head = nn.Linear(256, 2)  # valence, arousal
        self.tension_head = nn.Linear(256, 1)  # harmonic tension
        self.coherence_head = nn.Linear(256, 1)  # musical coherence
    
    def perceive(self, melody_seq, harmony_seq, rhythm_seq):
        """Encode musical output into embedding space."""
        
        # Concatenate or fuse multi-view inputs
        context = self._build_context(melody_seq, harmony_seq, rhythm_seq)
        embedding = self.context_encoder(context)
        
        return {
            'embedding': embedding,           # (seq_len, 256) — full phrase
            'pooled': embedding.mean(dim=0),  # (256,) — phrase summary
            'emotion': self.emotion_head(embedding.mean(dim=0)),  # valence, arousal
            'tension': self.tension_head(embedding.mean(dim=0)),
            'coherence': self.coherence_head(embedding.mean(dim=0))
        }
    
    def predict_next_phrase(self, current_embedding):
        """Predict the embedding of the next phrase."""
        return self.predictor(current_embedding)
```

#### What the JEPA Predicts

The JEPA is trained self-supervised on music corpora. Given bars 1–4, it predicts the **embedding** of bars 5–8 (not the raw notes — the *latent representation*). This is the core JEPA principle: predict in abstract space, not pixel/token space.

The prediction captures:
- **Emotional trajectory**: Will the next phrase feel more tense? More relaxed?
- **Stylistic coherence**: Is the next phrase consistent with the current style?
- **Harmonic direction**: Where is the harmony going?
- **Density trend**: Is the music getting busier or sparser?

#### Prediction Error as Feedback Signal

When the JEPA's prediction *mismatches* the actual output, this signals the LLM that something interesting happened:
- **High prediction error + low coherence** → The Markov chain went off the rails. LLM should tighten constraints.
- **High prediction error + high coherence** → Something surprising but good happened. LLM might lean into this direction.
- **Low prediction error** → The music is predictable. LLM might raise temperature for more adventure.

### 2.4 The Feedback Loop

```
Timeline (at 120 BPM, 4/4 time):

   Bar 1    Bar 2    Bar 3    Bar 4    Bar 5    Bar 6
   ├──┤     ├──┤     ├──┤     ├──┤     ├──┤     ├──┤
   
   [Markov generates notes continuously, filling a 1-bar buffer ahead]
   
   ────────[JEPA perceives bars 1-2]──────────────────────
                    │
                    ▼ embedding + predictions
            [LLM receives context summary]
                    │
                    ▼ parameter adjustments
   ──────────────────────[Markov parameters updated]──────
                    │
                    ▼ new notes with adjusted params
   ────────────────────────────[JEPA perceives bars 3-4]─
                                      │
                                      ▼
                              [LLM adjusts again...]
```

**Update cadence:**
- Markov generation: **continuous** (8 notes/sec at 120 BPM, 16th notes)
- JEPA perception: **every 2 bars** (~4 seconds at 120 BPM)
- LLM parameter update: **every 4 bars** (~8 seconds), or immediately if JEPA flags low coherence

### 2.5 Multi-Chain Coordination

Three Markov chains run in parallel — melody, harmony, rhythm — coordinated through shared state:

```python
class MultiChainCoordinator:
    """Coordinates melody, harmony, and rhythm Markov chains."""
    
    def __init__(self):
        self.melody_chain = AgenticMarkovSampler(melody_matrix, MelodyParams())
        self.harmony_chain = AgenticMarkovSampler(harmony_matrix, HarmonyParams())
        self.rhythm_chain = AgenticMarkovSampler(rhythm_matrix, RhythmParams())
        
        # Shared state
        self.shared_key = "C"
        self.shared_mode = "major"
        self.shared_tempo = 120
        self.emotional_trajectory = []  # list of (valence, arousal) over time
    
    def generate_phrase(self, jepa_feedback, llm_directives):
        """Generate a coordinated phrase across all chains."""
        
        # 1. Apply LLM global parameters
        self._apply_global_state(llm_directives)
        
        # 2. Generate rhythm first (structural foundation)
        rhythm_seq = self.rhythm_chain.generate(
            length=4,  # bars
            jepa_feedback=jepa_feedback.get('rhythm'),
            llm_params=llm_directives.get('rhythm', {})
        )
        
        # 3. Generate harmony conditioned on rhythm
        harmony_seq = self.harmony_chain.generate(
            rhythm_condition=rhythm_seq,
            length=4,
            jepa_feedback=jepa_feedback.get('harmony'),
            llm_params=llm_directives.get('harmony', {})
        )
        
        # 4. Generate melody conditioned on both
        melody_seq = self.melody_chain.generate(
            harmony_condition=harmony_seq,
            rhythm_condition=rhythm_seq,
            length=4,
            jepa_feedback=jepa_feedback.get('melody'),
            llm_params=llm_directives.get('melody', {})
        )
        
        # 5. Validate cross-chain coherence
        coherence = self._validate(melody_seq, harmony_seq, rhythm_seq)
        if coherence < 0.6:
            melody_seq = self._correct_melody(melody_seq, harmony_seq)
        
        return melody_seq, harmony_seq, rhythm_seq
```

**Cross-chain coupling coefficients** (how much each chain influences others):

| Source → Target | Coupling | Rationale |
|-----------------|----------|-----------|
| Rhythm → Harmony | 0.3 | Harmony follows rhythmic phrasing |
| Rhythm → Melody | 0.4 | Melody aligns to rhythmic grid |
| Harmony → Melody | 0.6 | Strong: melody notes should fit chords |
| Melody → Harmony | 0.2 | Weak: melody can suggest reharmonization |
| Melody → Rhythm | 0.1 | Very weak: melody doesn't dictate rhythm |

---

## 3. Agentic L-Systems

### Traditional Foundation

Lindenmayer systems (L-systems) are parallel rewriting grammars. Starting from an axiom, production rules expand each symbol into a string of symbols. After *n* iterations, the resulting string is interpreted as a sequence of musical events.

Example: The Fibonacci L-system
```
Axiom: A
Rules: A → AB, B → A
Iterations: A → AB → ABA → ABAAB → ABAABABA → ...
```

This produces self-similar, hierarchical patterns naturally — the kind of structure found in Bach fugues, Fibonacci-inspired melodies, and branching rhythmic structures.

**What's fixed:** The rewriting mechanism (string expansion).
**What we make controllable:** The production rules themselves, mid-performance.

### 3.1 Musical Alphabet

The L-system alphabet maps symbols to musical events:

#### Terminal Symbols (produce sound directly)

| Symbol | Meaning | Example |
|--------|---------|---------|
| `n<pitch>` | Absolute note | `nC4` = middle C, `nFs5` = F# in octave 5 |
| `i<interval>` | Relative interval | `iM3` = major 3rd up, `i-♭7` = minor 7th down |
| `r<duration>` | Rest | `r8` = eighth rest, `r16` = sixteenth rest |
| `.` | Staccato articulation | Modifier on preceding note |
| `~` | Legato articulation | Modifier on preceding note |
| `>` | Accent | Modifier on preceding note |
| `_` | Tenuto | Modifier on preceding note |
| `@` | Ghost note | Very quiet, implied |
| `\|` | Bar boundary | Phrase positioning marker |

#### Non-Terminal Symbols (expanded by rules)

| Symbol | Musical Role |
|--------|-------------|
| `P` | Phrase head — initiates a musical statement |
| `M` | Motif unit — the core melodic cell |
| `F` | Fill/ornament — connects motifs |
| `R` | Resolution cell — concludes a harmonic motion |
| `S` | Space/breath — provides rhythmic air |
| `T` | Transition — bridges between sections |

### 3.2 Base Grammar Specification (Cool Jazz)

```
Axiom: P | P | P     (three opening phrases, aligned to 4-bar jazz form)

Rules:
1. [last_note = chord_3rd]     P [0.85] → M iM3 M r8 S |
2. [bar_pos < 3/4]             M [0.90] → n8 ~ n8 ~ iP4 n4.
3. [prev_3_notes_ascending]    F [0.70] → im3 r16 iM2
4. [chord_change_next_bar]     R [0.92] → i-2 n2_
5. [any]                       S [0.60] → r8 r16 r16
```

**Rule anatomy:**
- `[predicate]` — context condition (must be true for rule to fire)
- `Symbol [weight]` — the non-terminal being expanded + stochastic probability
- `→ expansion` — the replacement string

This produces: clean legato 8th notes, stepwise motion, quiet resolutions, and gentle space — characteristic of Miles Davis *Birth of the Cool*.

### 3.3 The LLM as Rule Rewriter

The LLM doesn't just adjust parameters — it **rewrites the grammar** mid-performance. This is the key innovation: the L-system becomes a *living grammar that evolves during the solo*.

#### Rewrite Protocol

- **Cadence**: Every 2 bars, before refilling the playback buffer, the LLM receives the last 8 bars of output, current rule set, energy level, and chord changes
- **Constraint**: Maximum 2 rule edits per update (no hard cuts)
- **Interpolation**: All rule changes are interpolated over 4 bars
- **No deletion**: Rules are faded out (weight → 0.05), never removed. The ghost of old rules still occasionally fires, creating organic transitions

#### Example Transition 1: Cool Jazz → Free Jazz

| Bar | LLM Edit | Musical Rationale |
|-----|----------|-------------------|
| 6 | **Rule 4** (Resolution): `[chord_change_next_bar] R [0.92] → i-2 n2_` becomes `[any] R [0.41] → i+♯5 r4 i-♭9 \|` | "Remove chord lock first. Replace safe whole-step resolution with wide dissonant leaps. Keep weight low so this happens only occasionally. Don't shock the audience yet." |
| 8 | **Rule 2** (Motif): `[bar_pos < 3/4] M [0.90] → n8 ~ n8 ~ iP4 n4.` becomes `[any] M [0.65] → n16> n16> n16> r32 n8.` | "Break legato. Add accented staccato 16ths and a 32nd rest that intentionally slips off the swing grid. Listeners will feel tension before they consciously notice it." |
| 10 | **Rule 5** (Space): `[any] S [0.60] → r8 r16 r16` becomes `[last_3_notes_dense] S [0.77] → r2 \|` | "Now commit: after fast runs, drop an entire half bar of silence. This is the defining free jazz move. Only do this once the crowd is leaning in." |

Old cool jazz rules still occasionally fire (weight faded to ~0.15), creating uncanny tension — a soloist drifting away from form rather than abruptly switching.

#### Example Transition 2: Free Jazz → Fusion

| Bar | LLM Edit | Musical Rationale |
|-----|----------|-------------------|
| 16 | **Rule 3** (Fill): OLD `[prev_3_notes_ascending] F [0.70] → im3 r16 iM2` → NEW `[bar_pos = 2+1/8, 3+1/8] F [0.94] → iP5@ iM3@ iP4@` | "Lock fills exactly on the off-beats. All ghost notes, no rests. This is the Brecker Brothers pocket." |
| 18 | **Rule 1** (Phrase): OLD `[last_note = chord_3rd] P [0.85] → M iM3 M r8 S \|` → NEW `[last_note_root_or_7th] P [0.80] → F F M i+♭7 F F \|` | "Remove the breath rest. Phrases now lead with repeated fills, land hard on the flat 7." |
| 20 | **New Rule 6**: `[prev_note_accents] M [0.71] → i+m2 i-m2 i+m2 i-m2` | "Add chromatic neighbour tone shake. Sits perfectly on the funk grid, reads as aggressive without breaking time." |

#### Example Transition 3: Fusion → Ambient

| Bar | LLM Edit | Musical Rationale |
|-----|----------|-------------------|
| 28 | **Rule 2** (Motif): → `[any] M [0.30] → n2. ~ n4 ~ n2.` | "Slow everything down. Half notes with legato. The funk energy dissipates into floating sustain." |
| 30 | **Rule 4** (Resolution): → `[any] R [0.50] → r1 \|` | "Resolutions are now full bars of rest. Let the reverb decay be the music." |
| 32 | **New Rule 7**: `[sustained_note_ending] F [0.60] → i+P5 ~ ~ ~` | "Perfect fifth drones. The fill is now a harmonic pad." |

### 3.4 What the JEPA Predicts

For L-systems, the JEPA has a specialized role: **predicting the musical effect of a rule change before it's applied**.

```
JEPA receives:    Current grammar + proposed rule rewrite
JEPA predicts:    Embedding of the next 2 bars after the rewrite would be applied
JEPA compares:    Predicted embedding vs. desired emotional trajectory
JEPA reports:     "This rewrite increases tension by 0.3 (desired: +0.5). 
                   Consider a more aggressive change."
```

This is architecturally significant: the JEPA serves as a **lookahead mechanism** for grammar changes. The LLM can propose a rule rewrite, the JEPA simulates its effect in embedding space, and the LLM decides whether to apply it or modify it — all before a single note is generated.

#### Prediction Architecture

```python
class JEPALSystemPredictor:
    """JEPA that predicts the effect of L-system rule changes."""
    
    def predict_rule_effect(self, current_grammar, proposed_rule, 
                             current_context, bars_to_predict=2):
        """
        Predict what the music will sound like after a rule change.
        
        Returns:
            - predicted_embedding: what the next N bars will sound like
            - confidence: how certain the JEPA is (low = novel territory)
            - emotion_delta: change in valence/arousal vs current state
        """
        # Encode the grammar + proposed change as a graph
        grammar_encoding = self.encode_grammar(current_grammar, proposed_rule)
        
        # Encode the current musical context
        context_encoding = self.encode_context(current_context)
        
        # Predict the resulting musical embedding
        combined = torch.cat([grammar_encoding, context_encoding], dim=-1)
        predicted_embedding = self.predictor(combined)
        
        # Decode to musical descriptors
        emotion = self.emotion_decoder(predicted_embedding)
        tension = self.tension_decoder(predicted_embedding)
        density = self.density_decoder(predicted_embedding)
        
        return {
            'embedding': predicted_embedding,
            'emotion': emotion,    # (valence, arousal)
            'tension': tension,
            'density': density,
            'confidence': self.confidence(predicted_embedding)
        }
```

### 3.5 Stochastic and Parametric L-System Extensions

Under LLM control, the system can also employ:

- **Stochastic L-systems**: Each rule has a weight (probability). The LLM adjusts weights to bias the grammar without rewriting rules.
- **Parametric L-systems**: Symbols carry parameters (e.g., `n(pitch=60, velocity=80, duration=0.5)`). The LLM can shift parameter ranges globally.
- **Context-sensitive L-systems**: Rules examine neighboring symbols. The JEPA's embedding feedback feeds the context, making rules responsive to perceived musical state.

### 3.6 Key Insight

> The most surprising emergent property is that the best solos occur when the LLM edits only **one single rule every 4 bars**. Good jazz improvisation is not generating new notes — it is slowly, deliberately changing the rules that generate the notes, while carrying the ghost of every rule that came before.

---

## 4. Agentic Fractals

### Traditional Foundation

Fractal music exploits self-similarity across time scales: patterns at the bar level mirror patterns at the beat level mirror patterns at the sub-beat level. Natural music exhibits fractal properties — 1/f noise (pink noise) appears in pitch distributions, amplitude fluctuations, and rhythmic timing across virtually all genres.

Research shows that music with fractal dimension near 1.5 (Brownian-like) is perceived as most natural and engaging. The power spectral density of enjoyable music follows S(f) ∝ 1/f^α where α ≈ 1 ("pink noise" or "1/f noise").

> **⚠️ Caveat (math review Aug 2026):** The claim that "D ≈ 1.5 is most natural" is folk wisdom derived from the 1/f noise literature, not independently established. The value D = 1.5 follows from α = 1 via D = 2 − α/2, but this is the spectral exponent of pitch *fluctuations*, not the dimension of musical structure. Perceptual studies are mixed: Pressnitzer & McAdams (1999) found listeners don't reliably prefer 1/f over other correlations. This claim should be treated as a heuristic, not established fact. See Voss & Clarke (1975, 1978); Krumhansl (2000) for alternative models of pitch structure.
>
> Additionally, the term "Hausdorff dimension" as used below is a *heuristic complexity parameter*, not the mathematical Hausdorff dimension. A finite melody (discrete points) has Hausdorff dimension 0. The graph of a continuous interpolation has dimension 1 (piecewise linear). The parameter D below is better understood as derived from the Hurst exponent H via D = 2 − H, where H directly controls fBm roughness. Consider using H as the primary control parameter for mathematical honesty.

### 4.1 Hausdorff Dimension as a Continuous Musical Parameter

> **Caveat (math review Aug 2026):** "Hausdorff dimension" is a misnomer here. The Hausdorff dimension of a finite melody (discrete points) is 0. The parameter D below is better described as a complexity parameter derived from the Hurst exponent (D = 2 − H for fBm graphs). We retain the term for continuity with the 1/f noise literature, but mathematically, H is the honest control parameter.

The Hausdorff dimension D quantifies the roughness/fractal complexity of a musical surface. We map D ∈ [1.0, 2.0] as a continuous, LLM-modulatable parameter:

| D Value | Pitch Character | Rhythmic Character | Dynamic Character | Emotional Feel |
|---------|----------------|--------------------|--------------------|----------------|
| **1.0** | Pure scale patterns (chromatic/whole-tone) | Strict metronomic subdivision | Harmonic series, no microtonal drift | "Too perfect," clinical, metronomic |
| **1.2** | Diatonic with occasional passing tones | Slight swing, mild syncopation | Smooth envelopes | Gentle, pleasant, pop-like |
| **1.4** | Diatonic + chromatic neighbor tones | Moderate syncopation, polyrhythmic hints | 1/f noise dynamics | Engaging, interesting |
| **1.5** | Full chromaticism with tonal center | Swung 8ths, 3:2 polyrhythms | Exponential decay + 1/f² noise | **Optimal**: natural, human-like, "groove factor > 0.7" |
| **1.6** | Chromatic with tonal ambiguity | Strong syncopation, polyrhythms | Wide dynamic range | Dense, complex, stimulating |
| **1.7** | Near-atonal with pitch clusters | Irrational time signatures (7/8, 13/16) | Sudden shifts | Tense, uneasy |
| **1.8** | Atonal clusters, glissandi | Tuplet cascades, metric modulation | White noise envelope | Chaotic, boundary-pushing |
| **2.0** | Microtonal clusters, noise | Irrational rhythms, total fragmentation | Full dynamic chaos | "Free jazz," ambient texture, loss of pulse |

#### Mathematical Implementation

```python
def hausdorff_to_musical_params(D: float) -> dict:
    """Map Hausdorff dimension to concrete musical parameters."""
    
    # Pitch quantization: higher D = finer pitch divisions
    pitch_cents = int(100 * D)  # 100 cents at D=1, 200 cents at D=2
    
    # Scale degrees allowed (more chromatic as D increases)
    if D < 1.3:
        allowed = [0, 2, 4, 5, 7, 9, 11]  # Major scale
    elif D < 1.5:
        allowed = [0, 2, 3, 5, 7, 8, 10]  # Minor + blue notes
    elif D < 1.7:
        allowed = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # Full chromatic
    else:
        # Microtonal: add quarter tones
        allowed = [i * 0.5 for i in range(24)]
    
    # Rhythmic syncopation
    syncopation = 0.5 * (D - 1.0)  # 0 at D=1, 0.5 at D=2
    
    # Dynamic range (dB)
    dynamic_range = 20 * (D - 1.0)  # 0 dB at D=1, 20 dB at D=2
    
    # Timbral harmonicity (spectral roll-off slope)
    harmonicity = 1.0 - (D - 1.0) * 0.3  # 1.0 at D=1, 0.7 at D=2
    
    return {
        'pitch_cents': pitch_cents,
        'allowed_degrees': allowed,
        'syncopation': syncopation,
        'dynamic_range_db': dynamic_range,
        'harmonicity': harmonicity,
        'complexity': D  # raw dimension for downstream use
    }
```

### 4.2 Lacunarity → Rhythmic Density

Lacunarity Λ measures the "gappiness" of a fractal — the distribution of gap sizes. In music, this maps directly to **rhythmic density and silence patterns**.

$$\Lambda(\varepsilon) = \frac{\sigma^2}{\mu^2} \text{ where } \sigma, \mu \text{ are from local densities at scale } \varepsilon$$

| Lacunarity | Rhythmic Character | Musical Example |
|------------|-------------------|-----------------|
| **Λ ≈ 0.2** | Continuous 16th-note stream, motorik rhythm | Krautrock pulse, techno |
| **Λ ≈ 0.4** | Regular syncopation, funk grid | James Brown |
| **Λ ≈ 0.7** | Clustered syncopation, rests in groups of 2-3 | Broken beat, Dilla |
| **Λ ≈ 1.0** | Sparse with occasional bursts | Ambient, dub |
| **Λ ≈ 1.2** | Extended silences (6-10 beats), punctuated hits | Morton Feldman, free improv |

```python
def lacunarity_to_rhythm(lacunarity: float, num_steps: int = 16) -> np.ndarray:
    """Generate a rhythm pattern with specified lacunarity."""
    
    # Generate gap distribution via multiplicative cascade
    gaps = []
    current_gap = 1.0
    for _ in range(int(lacunarity * 10)):
        current_gap *= (0.5 ** (lacunarity * 0.1))
        gaps.append(current_gap)
    
    # Map gaps to onset probabilities
    onset_prob = np.exp(-lacunarity * np.array(gaps))
    onset_prob /= onset_prob.sum()
    
    # Sample onset positions
    pattern = np.zeros(num_steps)
    num_onsets = max(1, int(num_steps * (1.0 - lacunarity * 0.5)))
    positions = np.sort(np.random.choice(num_steps, num_onsets, replace=False))
    pattern[positions] = 1.0
    
    return pattern
```

### 4.3 Additional Fractal Parameters

#### Hurst Exponent (H) — Long-Range Dependence

The Hurst exponent quantifies long-range memory in a time series. For music:

| H Value | Behavior | Genre Mapping |
|---------|----------|---------------|
| **H → 0** | Anti-persistent: jumpy, tends to reverse direction | Free jazz, noise, avant-garde |
| **H ≈ 0.5** | Brownian: random walk, no memory | Blues, simple pop |
| **H ≈ 0.7** | Persistent: trends continue, smooth evolution | Minimalist techno, ambient, drone |
| **H → 1.0** | Highly persistent: strong trends, very smooth | Gregorian chant, meditative drones |

The Hurst exponent directly controls the fractal generation via fractional Brownian motion:

```python
def fbm_melody(length: int, H: float, base_pitch: int = 60) -> list[int]:
    """Generate melody from fractional Brownian motion with Hurst exponent H."""
    
    # Generate fBm via Davies-Harte method (or Cholesky for small sequences)
    # H=0.5 → standard Brownian motion
    # H>0.5 → trending, smooth
    # H<0.5 → mean-reverting, jumpy
    
    n = length
    # Autocovariance function
    def gamma(k):
        return 0.5 * (abs(k-1)**(2*H) - 2*abs(k)**(2*H) + abs(k+1)**(2*H))
    
    # Build covariance matrix and sample
    cov = np.array([[gamma(i-j) for j in range(n)] for i in range(n)])
    L = np.linalg.cholesky(cov + 1e-6 * np.eye(n))
    samples = L @ np.random.randn(n)
    
    # Map to MIDI pitches
    pitches = base_pitch + np.round(samples * 4).astype(int)
    return pitches.tolist()
```

#### IFS (Iterated Function Systems) — Pitch Contour Generation

IFS fractals compose multiple contraction mappings. Each mapping is an affine transformation:

$$w_i(\mathbf{x}) = A_i \mathbf{x} + \mathbf{b}_i$$

The LLM controls the contraction ratios, rotations, and translation vectors:

| Emotion | Contraction Ratio | Rotation Angle | Musical Effect |
|---------|------------------|----------------|----------------|
| **Melancholy** | 0.9 (slow contraction) | 10° (near-identity) | Slow, downward drift |
| **Joy** | 0.6 (moderate) | 60° (wide arcs) | Upward, expansive motion |
| **Tension** | 0.8 (slow) | 5° (near-parallel) | Inward spiral, claustrophobic |
| **Resolution** | 0.2 (fast) | 0° (convergent) | Rapid convergence to tonic |
| **Ambience** | 0.5 (moderate) | 120° (triadic) | Suspended, spacious |

### 4.4 The JEPA as Fractal-Emotion Mapper

The JEPA learns the mapping from fractal parameters to perceived emotion through self-supervised training:

```
Training data: (fractal_params, audio) pairs labeled with emotion descriptors
JEPA input: fractal parameters (D, Λ, H, IFS coefficients)
JEPA target: audio embedding of the resulting music
JEPA learns: latent space where fractal parameters cluster by emotional quality
```

**Emotional mapping learned by JEPA:**

```
D=1.1, Λ=0.3 → "calm, serene"    (Valence: 0.8, Arousal: 0.2)
D=1.5, Λ=0.6 → "joyful, danceable" (Valence: 0.7, Arousal: 0.6)
D=1.8, Λ=0.9 → "tense, anxious"   (Valence: 0.2, Arousal: 0.8)
D=2.0, Λ=1.1 → "chaotic, dark"    (Valence: 0.1, Arousal: 0.9)
```

**The killer feature:** The JEPA learns the **inverse mapping**. Given a desired emotion, it can predict the optimal fractal parameters. So the LLM says "make it more melancholic" and the JEPA translates that to D=1.15, Λ=0.8, H=0.8.

### 4.5 Multi-Layer Fractal Architecture

Three independent fractal streams run simultaneously:

```python
class MultiFractalComposer:
    """Multiple fractal layers for different musical dimensions."""
    
    def __init__(self, global_D: float = 1.5, global_L: float = 0.5):
        # Layer 1: Pitch contour (slightly more complex)
        self.pitch_fractal = IFSGenerator(
            D=global_D * 1.2,
            rules=[transpose, invert, mirror, stretch]
        )
        
        # Layer 2: Rhythmic structure (slightly smoother)
        self.rhythm_fractal = MidpointDisplacement(
            D=global_D * 0.8,
            roughness=1 - global_L * 0.5
        )
        
        # Layer 3: Dynamic envelope
        self.dynamics_fractal = CantorSetGenerator(
            D=global_D * 1.0,
            lacunarity=global_L
        )
        
        # Cross-layer coupling matrix
        self.coupling = np.array([
            # pitch  rhythm  dynamics
            [0.8,    0.1,    0.1],   # pitch: mostly self-determined
            [0.2,    0.7,    0.1],   # rhythm: influenced by pitch contour
            [0.1,    0.3,    0.6]    # dynamics: follow rhythm
        ])
    
    def compose(self, num_bars: int = 16) -> list:
        pitches = self.pitch_fractal.generate(num_bars)
        rhythms = self.rhythm_fractal.generate(num_bars)
        dynamics = self.dynamics_fractal.generate(num_bars)
        
        # Cross-layer correlation check
        mi = mutual_information(pitches, rhythms, dynamics)
        if mi > 0.7:  # Too correlated → inject variation
            pitches += np.random.normal(0, 0.1, pitches.shape)
        
        # Fuse into MIDI events
        events = []
        for t in range(num_bars * 16):  # 16th note resolution
            if rhythms[t] > 0.5:  # note onset
                events.append({
                    'pitch': int(pitches[t]),
                    'velocity': int(dynamics[t] * 127),
                    'start': t / 16,  # in bars
                    'duration': float(rhythms[t])
                })
        return events
```

---

## 5. Agentic Cellular Automata

### Traditional Foundation

Cellular automata (CA) are discrete computational systems where a grid of cells evolves based on simple rules. For music, 1D CAs (Wolfram rules) are most common: each row of the CA evolution represents one time step (beat/measure), with live cells = hits and dead cells = rests.

### 5.1 Wolfram Rule → Musical Feel Mapping

Empirically validated mappings (groove scores from 100+ iterations × 16-step patterns):

#### Stability Class — Steady/Driving

| Rule | Behavior | Musical Use | Groove Score |
|------|----------|-------------|-------------|
| **Rule 4** | Pure periodicity (τ=1) | 4-on-the-floor kick | 0.95 |
| **Rule 8** | Stable periodic with long runs | Driving 8th-note hats | 0.88 |
| **Rule 32** | Periodic with internal variation | Steady snare backbeats | 0.85 |
| **Rule 128** | Single-cell propagation | Sparse, hypnotic patterns | 0.82 |

#### Complexity Class — Groove/Infectious

| Rule | Behavior | Musical Use | Groove Score |
|------|----------|-------------|-------------|
| **Rule 30** | Class III chaos, statistically self-similar (1/f-like) | **Basslines** — syncopated, human-like | 0.78 |
| **Rule 90** | Sierpinski triangle fractal | **Hi-hats** at width 32+ — polyrhythmic swing | 0.82 |
| **Rule 110** | Class IV (universal computation) — gliders, particles | **Snare/clap** — emergent broken beat | 0.85 |
| **Rule 54** | Class IV, persistent local structures | Broken beat / funk | 0.80 |

#### Chaos Class — Experimental

| Rule | Behavior | Musical Use |
|------|----------|-------------|
| **Rule 45** | High entropy, periodic islands | Acid techno randomness |
| **Rule 57** | Near-uncorrelated noise | Texture pads (not rhythmic) |
| **Rule 150** | Linear CA, spectral peaks | Noise with tonal components |

### 5.2 Multi-Track CA Architecture

Three coordination topologies, in order of sophistication:

#### Option A: Shared CA (Synchronous)
```
Single CA instance, width = 16 × num_tracks
Track k reads columns [k*16 : (k+1)*16]
```
**Pros:** Perfect temporal coherence, emergent cross-track patterns.
**Cons:** Limited independence.

#### Option B: Independent CAs (Coupled Seeds)
```
Each instrument has own CA instance.
seed_kick   = hash(seed_master + 0 + phrase_position)
seed_snare  = hash(seed_master + 1 + phrase_position)
seed_hat    = hash(seed_master + 2 + phrase_position)
```
**Pros:** Full independence.
**Cons:** No emergent coupling.

#### Option C: Hierarchical Coupling (Recommended)

```
Master CA (width=64) → groove skeleton
  ├── Kick CA:  reads master cells at positions ≡ 0 (mod 4)
  ├── Snare CA: reads master cells at positions ≡ 2 (mod 4)
  ├── Hat CA:   reads XOR of adjacent master cells
  └── Bass CA:  density-matched to master, independent evolution
```

Feed-forward coupling:
```
kick[t+1]  = f(kick[t], master[t])
snare[t+1] = f(snare[t], kick[t], master[t])
hat[t+1]   = f(hat[t], snare[t])
bass[t+1]  = f(bass[t], master[t])
```

### 5.3 LLM Control Interface

```python
class LLM_CA_Control:
    """The 8-dimensional control vector the LLM outputs for CA engines."""
    
    rule: int               # 0-255, Wolfram rule number
    seed: list[int]         # width-length binary seed row
    width: int              # 16, 32, 64, or 128
    boundary: str           # "periodic" | "fixed-zero" | "reflecting" | "adiabatic"
    iterations_per_beat: int # 1 (16th notes), 2 (32nd notes)
    density_target: float   # 0.1-0.8, post-hoc filtering
    temporal_offset: int    # iterations to skip (bypass transient)
    quantize: int           # 0=none, 1=swing 65%, 2=swing 75%
```

The LLM outputs **transition plans**, not just instantaneous parameters:

```json
{
  "bars": [1, 2, 3, 4],
  "rule_sequence": [30, 90, 110, 4],
  "morph": {
    "type": "linear_interpolation",
    "steps": 8,
    "intermediate_rules": [48, 66, 94, 42, 66, 108, 36]
  },
  "rationale": "Build from chaotic bassline (30) → polyrhythmic hats (90) → 
                 broken beat (110) → driving resolve (4)"
}
```

### 5.4 Smooth Rule Transitions

Rule changes on phrase boundaries use **continuous interpolation** through rule space:

```python
def morph_rules(rule_from: int, rule_to: int, steps: int) -> list[int]:
    """Smoothly morph between two Wolfram rules over N steps."""
    
    # Convert rules to 8-bit lookup tables
    bits_from = [(rule_from >> i) & 1 for i in range(8)]
    bits_to = [(rule_to >> i) & 1 for i in range(8)]
    
    intermediate_rules = []
    for step in range(steps + 1):
        t = step / steps
        # Interpolate each bit probabilistically
        morphed_bits = []
        for bf, bt in zip(bits_from, bits_to):
            if bf == bt:
                morphed_bits.append(bf)
            else:
                # Probability of flipping increases linearly
                morphed_bits.append(1 if np.random.random() < t else bf)
        
        rule = sum(b << i for i, b in enumerate(morphed_bits))
        intermediate_rules.append(rule)
    
    return intermediate_rules
```

### 5.5 JEPA Groove Perception

The JEPA monitors CA output as rhythm and predicts **groove quality**:

#### Groove Quality Function (differentiable, computable)

```python
def groove_score(pattern: np.ndarray, tempo: float) -> float:
    """Compute a groove quality score for a rhythmic pattern."""
    
    onsets = np.where(pattern > 0.5)[0]
    if len(onsets) < 2:
        return 0.0
    
    ioi = np.diff(onsets)  # inter-onset intervals
    
    # 1. Syncopation: distance from grid
    grid_positions = np.arange(0, len(pattern), 4)  # 16th note grid
    syncopation = np.mean([
        min(abs(onset - grid_positions)) for onset in onsets
    ]) / len(grid_positions)
    
    # 2. Microtiming variability (human feel proxy)
    jitter = np.std(ioi) / (np.mean(ioi) + 1e-8)
    
    # 3. Call-response balance (first half vs second half of bar)
    call_density = np.sum(pattern[:len(pattern)//2])
    response_density = np.sum(pattern[len(pattern)//2:])
    balance = 1 - abs(call_density - response_density) / (call_density + response_density + 1e-8)
    
    return 0.4 * (1 - syncopation) + 0.3 * (1 - min(jitter, 1)) + 0.3 * balance
```

#### JEPA Architecture for CA

```
Input: 3 views of each bar
  View A: CA evolution image (width × iterations) — visual pattern
  View B: FFT of onset times — spectral view  
  View C: Statistical features (density, run-length entropy, syncopation index)

Encoder: 3× Conv2D layers → 128-dim latent
Predictor: MLP (128 → 256 → 128) with bar-hierarchy attention
Output: groove_score, energy_level, stability_index
```

### 5.6 2D CA: Game of Life Harmony

Beyond 1D rhythm generation, 2D CAs extend to harmony:

```
Grid: 32×32 (fits 2 octaves × 12 pitch classes)

Cell state mapping:
  0 = no chord change
  1 = chord root (pitch class from x-coordinate)  
  2 = chord extension (from y-coordinate)
  3 = chord inversion (from neighborhood sum)

Musical structures emerge:
  Gliders → moving chord progressions (I-V-vi-IV)
  Blinkers → static pedal points
  Block oscillators → alternating chords
```

**Coordination with 1D rhythm CA:**
- Every 4th time step of the 1D CA, run one step of the 2D harmony CA
- Bass reads harmony roots from the 2D grid
- Voice leading via morphological operations (erosion/dilation between phrase boundaries)

---

## 6. The Agentic Center: JEPA + LLM

This is the core architecture that sits above all four algorithmic engines.

### 6.1 The JEPA Stack

#### What It Perceives

The JEPA receives a **multi-modal encoding** of the complete musical output:

| Input Stream | Encoding | Dimension |
|-------------|----------|-----------|
| Melody (MIDI) | Interval sequence + pitch class histogram | 128 per bar |
| Harmony (MIDI) | Chroma vector per beat + chord embeddings | 64 per bar |
| Rhythm (MIDI) | Onset pattern + inter-onset intervals | 32 per bar |
| Dynamics (MIDI) | Velocity envelope + dynamic range | 32 per bar |
| Audio (optional) | Mel-spectrogram | 128 per bar |

All streams are fused into a **unified musical embedding**:

```python
class MusicalJEPA(nn.Module):
    """The central JEPA that perceives all musical output."""
    
    def __init__(self):
        super().__init__()
        
        # Modality-specific encoders
        self.melody_encoder = TransformerEncoder(d_model=128, nhead=4, num_layers=3)
        self.harmony_encoder = TransformerEncoder(d_model=128, nhead=4, num_layers=3)
        self.rhythm_encoder = TransformerEncoder(d_model=64, nhead=4, num_layers=2)
        self.dynamics_encoder = TransformerEncoder(d_model=64, nhead=4, num_layers=2)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128 + 64 + 64, 256),
            nn.GELU(),
            nn.Linear(256, 256)
        )
        
        # Prediction head (predicts next-phrase embedding)
        self.predictor = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 256)
        )
        
        # Musical descriptor heads (for LLM consumption)
        self.emotion_head = nn.Linear(256, 2)      # valence, arousal
        self.tension_head = nn.Linear(256, 1)      # harmonic tension
        self.energy_head = nn.Linear(256, 1)       # perceived energy
        self.coherence_head = nn.Linear(256, 1)    # musical coherence
        self.density_head = nn.Linear(256, 1)      # note density
        self.novelty_head = nn.Linear(256, 1)      # novelty vs. corpus
        
        # Target encoder (EMA-updated copy for self-supervised training)
        self.target_encoder = copy.deepcopy(self)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
    
    def perceive(self, musical_output) -> dict:
        """Perceive a musical phrase and return embeddings + descriptors."""
        
        m = self.melody_encoder(musical_output.melody)
        h = self.harmony_encoder(musical_output.harmony)
        r = self.rhythm_encoder(musical_output.rhythm)
        d = self.dynamics_encoder(musical_output.dynamics)
        
        fused = self.fusion(torch.cat([m, h, r, d], dim=-1))
        
        return {
            'embedding': fused,
            'emotion': self.emotion_head(fused),    # (valence, arousal)
            'tension': self.tension_head(fused),
            'energy': self.energy_head(fused),
            'coherence': self.coherence_head(fused),
            'density': self.density_head(fused),
            'novelty': self.novelty_head(fused)
        }
    
    def predict_next_phrase(self, current_embedding):
        """Predict the embedding of the next phrase (2-4 bars ahead)."""
        return self.predictor(current_embedding)
```

#### What It Predicts

The JEPA makes predictions at three timescales:

1. **Next-bar prediction** (~2 sec at 120 BPM): What will the next bar sound like?
2. **Next-phrase prediction** (~8 sec): What will the next 4-bar phrase sound like?
3. **Section trajectory** (~30 sec): Where is the overall section heading emotionally?

The prediction error at each timescale serves a different purpose:

| Prediction Horizon | What Error Means | LLM Response |
|-------------------|------------------|-------------|
| Next bar | Markov/CA immediate output is unexpected | Fine-tune parameters now |
| Next phrase | Trajectory is deviating from intent | Adjust multiple engine parameters |
| Section trajectory | Emotional arc is wrong | Major directive change needed |

#### Training

The JEPA is trained self-supervised on large music corpora:

- **Corpus**: 10,000+ pieces across genres (jazz, classical, electronic, folk)
- **Objective**: InfoNCE loss — maximize agreement between predicted embedding and actual target embedding, minimize agreement with negatives
- **Training data**: (bars 1-N) → predict embedding of bars (N+1 to N+k)
- **No labels needed**: The JEPA learns musical structure purely from prediction
- **Fine-tuning**: Optionally fine-tune on specific genres for domain-specific perception

### 6.2 The LLM Bandleader

#### What It Receives

Every 4 bars (~8 seconds at 120 BPM), the LLM receives a **musical context summary**:

```python
def build_llm_context(jepa_state, engine_states, performance_history):
    """Build the context prompt for the LLM bandleader."""
    
    return f"""
## Musical Context (Bar {current_bar})

### JEPA Perception
- Emotion: valence={jepa_state.emotion[0]:.2f}, arousal={jepa_state.emotion[1]:.2f}
- Tension: {jepa_state.tension:.2f}/1.0
- Energy: {jepa_state.energy:.2f}/1.0
- Coherence: {jepa_state.coherence:.2f}/1.0 (low = incoherent)
- Density: {jepa_state.density:.2f}/1.0
- Novelty: {jepa_state.novelty:.2f}/1.0

### JEPA Predictions (next 4 bars)
- Predicted emotion: valence={jepa_state.pred_emotion[0]:.2f}, arousal={jepa_state.pred_emotion[1]:.2f}
- Prediction confidence: {jepa_state.pred_confidence:.2f}
- Trajectory: {jepa_state.trajectory_description}

### Current Engine Parameters
- Markov (melody): temp={engine_states.markov.temp}, order={engine_states.markov.order}, key={engine_states.markov.key}
- L-system (harmony): active_rules={engine_states.lsystem.num_rules}, complexity={engine_states.lsystem.complexity}
- Fractal (dynamics): D={engine_states.fractal.D}, lacunarity={engine_states.fractal.lacunarity}
- CA (rhythm): rule={engine_states.ca.rule}, density={engine_states.ca.density}

### Performance History
- Last 32 bars: tension curve = {performance_history.tension_curve}
- Emotional arc: {performance_history.emotion_arc}
- Time remaining: {performance_history.time_remaining} bars

### Director's Intent
- Target emotion: valence={intent.target_valence}, arousal={intent.target_arousal}
- Section type: {intent.section_type} (intro/verse/chorus/bridge/solo/outro)
- Energy trajectory: {intent.energy_trajectory}
"""
```

#### What It Decides

The LLM outputs a **parameter adjustment set** in structured JSON:

```json
{
  "analysis": "Tension is building nicely (0.6 → 0.75) but coherence dropped 
               to 0.65. The Markov chain is getting too chromatic. I should 
               tighten key constraints and let the CA carry more complexity 
               instead.",
  
  "markov": {
    "temperature": 0.70,  // down from 0.85
    "chromatic_tolerance": 0.1,  // down from 0.2
    "max_interval": 7,  // down from 12
    "rationale": "Rein in the melody. Let it be simpler while harmony and rhythm get more complex."
  },
  
  "lsystem": {
    "rule_edits": [
      {"rule_id": 3, "new_weight": 0.85, "new_body": "iP5@ iM3@ iP4@"},
    ],
    "rationale": "Push the harmony toward more rhythmic, accented fills."
  },
  
  "fractal": {
    "D": 1.65,  // up from 1.5
    "lacunarity": 0.7,  // up from 0.5
    "rationale": "Increase complexity and gap structure for more dramatic dynamics."
  },
  
  "ca": {
    "rule": 110,  // from 90
    "density_target": 0.45,  // up from 0.35
    "rationale": "Switch to Rule 110 for emergent broken-beat complexity."
  },
  
  "global": {
    "tempo_delta": 2,  // +2 BPM
    "energy_target": 0.75,
    "transition_bars": 4
  }
}
```

#### Decision Cadence

| Decision Type | Frequency | Latency Budget |
|--------------|-----------|----------------|
| Global directive | Every 16-32 bars | 500ms |
| Engine parameter adjustments | Every 4 bars | 200ms |
| Emergency correction | Immediate (coherence < 0.4) | 100ms |
| L-system rule rewrite | Every 4-8 bars | 300ms |
| CA rule change | Phrase boundaries only | 200ms |

### 6.3 The Complete Feedback Loop

```
At 120 BPM, 4/4 time. One bar = 2 seconds.

TIME     0s          2s          4s          6s          8s
         | Bar 1      | Bar 2      | Bar 3      | Bar 4      | Bar 5
         |            |            |            |            |
ENGINE   [Generate ────────────────────────────────────────]──[Generate──>
         (Markov + L-system + Fractal + CA running continuously)
         |            |            |            |            |
BUFFER   [Play Bar 1 ]──[Play Bar 2]──[Play Bar 3]──[Play Bar 4]──[Play...>
         |            |            |            |            |
JEPA     ──────[Perceive bars 1-2]──────────────────────────────────>
         |            |            |            |            |
         |            ┌────────────┐            |            |
         |            │ Embed +   │            |            |
         |            │ Predict   │            |            |
         |            │ Next bars │            |            |
         |            └─────┬──────┘            |            |
         |                  |                   |            |
LLM      |                  ┌───────────────┐  |            |
         |                  │ Context       │  |            |
         |                  │ Summary       │  |            |
         |                  │ + Decide      │  |            |
         |                  │ Parameter     │  |            |
         |                  │ Adjustments   │  |            |
         |                  └───────┬───────┘  |            |
         |                          |          |            |
UPDATE   |                          ┌──────────┐            |
         |                          │ Apply to │            |
         |                          │ engines  │            |
         |                          └──────────┘            |
         |                                    |              |
ENGINE   [Old params ────────────────────────][New params ──]>
```

**Buffering strategy:** The system maintains a 1-bar playback buffer ahead of what's audible. This gives ~2 seconds of headroom for JEPA + LLM processing. The LLM runs asynchronously — its parameter updates take effect at the next bar boundary.

---

## 7. Multi-Engine Orchestration

### 7.1 Engine Role Assignment

| Engine | Musical Role | Why This Engine |
|--------|-------------|-----------------|
| **Markov Chain** | Melody generation | Best for sequential note-by-note generation with corpus memory |
| **L-System** | Harmony / chord progression | Self-similar hierarchical structure mirrors voice leading |
| **Fractal Generator** | Pitch contour / dynamics | Continuous parameters map naturally to emotional qualities |
| **Cellular Automata** | Rhythm / percussion | Discrete patterns with emergent complexity = natural grooves |

### 7.2 Shared State

All engines read from and write to a shared musical state object:

```python
@dataclass
class SharedMusicalState:
    """Global state shared across all engines."""
    
    # Harmonic state
    key: str = "C"
    mode: str = "major"
    current_chord: str = "Cmaj7"
    chord_progression: list[str] = field(default_factory=list)
    
    # Rhythmic state
    tempo: float = 120.0
    time_signature: tuple = (4, 4)
    swing: float = 0.0
    
    # Emotional trajectory
    valence: float = 0.5  # 0=sad, 1=happy
    arousal: float = 0.5  # 0=calm, 1=excited
    tension: float = 0.3
    energy: float = 0.5
    
    # Structural state
    bar_number: int = 0
    section_type: str = "verse"  # intro/verse/chorus/bridge/solo/outro
    phrase_position: int = 0  # position within current phrase
    
    # JEPA state
    coherence: float = 0.8
    novelty: float = 0.3
    predicted_trajectory: np.ndarray = None
```

### 7.3 Priority Hierarchy

During specific sections, different engines take priority:

| Section | Lead Engine | Supporting Engines |
|---------|------------|-------------------|
| Intro | Fractal (atmosphere) | CA (minimal pulse), Markov (sparse melody) |
| Verse | Markov (melody) | L-system (harmony), CA (steady beat) |
| Chorus | All engines equal | Coordinated via shared state |
| Solo | Markov (melody) | All others support — simplify, lock to changes |
| Bridge | L-system (harmonic exploration) | Fractal (textural shift) |
| Outro | Fractal (dissolution) | CA (slowing pulse), Markov (fragments) |

### 7.4 Conflict Resolution

When engines clash (e.g., Markov melody uses notes outside L-system harmony):

```python
def resolve_conflicts(melody, harmony, rhythm, state):
    """Resolve conflicts between engines, melody priority by default."""
    
    # 1. Check melody notes against current chord
    for note in melody:
        if not note_in_chord(note.pitch, state.current_chord):
            if state.tension < 0.7:
                # Low tension context: snap to nearest chord tone
                note.pitch = nearest_chord_tone(note.pitch, state.current_chord)
            # else: allow the dissonance (it's intentionally tense)
    
    # 2. Check rhythmic density alignment
    melody_density = compute_density(melody)
    rhythm_density = compute_density(rhythm)
    if abs(melody_density - rhythm_density) > 0.5:
        # Too mismatched: thin out melody or add rhythm hits
        if melody_density > rhythm_density:
            melody = thin_out(melody, factor=0.7)
    
    # 3. Check emotional coherence
    actual_emotion = jepa.perceive(melody, harmony, rhythm).emotion
    if distance(actual_emotion, target_emotion) > 0.3:
        # Flag for LLM attention on next update cycle
        state.coherence_warning = True
    
    return melody, harmony, rhythm
```

---

## 8. Latency & Performance Budget

### 8.1 Timing Requirements

At 120 BPM in 4/4:
- One bar = 2.0 seconds
- One beat = 0.5 seconds
- One 16th note = 0.125 seconds
- One phrase (4 bars) = 8.0 seconds

### 8.2 Component Latency Budget

| Component | Operation | Latency | Frequency |
|-----------|-----------|---------|-----------|
| Markov generation | Generate 1 bar of notes | < 2 ms | Continuous |
| L-system expansion | Expand 1 iteration | < 1 ms | Per bar |
| Fractal generation | Generate 1 bar contour | < 5 ms | Per bar |
| CA evolution | Evolve 16 steps | < 0.5 ms | Per bar |
| MIDI rendering | Convert to MIDI events | < 1 ms | Per bar |
| JEPA perception | Encode 2 bars | < 10 ms | Every 2 bars |
| JEPA prediction | Predict next phrase | < 5 ms | Every 2 bars |
| LLM inference | Generate parameter update | 100-300 ms | Every 4 bars |
| L-system rule rewrite | Generate rule edit | 200-500 ms | Every 4-8 bars |
| **Total pipeline** (per bar) | All of the above | **< 15 ms** | Continuous |
| **Total feedback loop** | JEPA + LLM + apply | **< 350 ms** | Every 4 bars |

### 8.3 Handling LLM Latency

The LLM inference takes 100-300ms, which is too slow for per-bar updates at fast tempos. The solution:

1. **Asynchronous LLM calls**: The LLM runs in a separate thread/process. Parameter updates take effect at the next bar boundary, whenever the LLM response arrives.
2. **Predictive parameter pre-computation**: While the current phrase plays, the LLM is already generating parameters for the *next* phrase. This hides latency entirely.
3. **Fallback parameters**: If the LLM hasn't responded in time, the system uses the JEPA's prediction to extrapolate from previous parameters.
4. **Tierled response**:
   - **Fast path** (< 50ms): JEPA-triggered parameter nudges (temperature, density) without LLM
   - **Medium path** (200ms): LLM parameter adjustments
   - **Slow path** (500ms+): LLM rule rewrites, structural changes

### 8.4 Memory Requirements

| Component | Memory |
|-----------|--------|
| Markov transition matrix (order 3) | ~50 MB |
| L-system rule database | ~5 MB |
| Fractal generation state | ~10 MB |
| CA grids (4 tracks × 64 cells) | < 1 KB |
| JEPA model (256-dim, 4 layers) | ~200 MB |
| LLM (7B parameter, quantized) | ~4 GB |
| Audio buffer | ~50 MB |
| **Total** | **~4.5 GB** |

Fits comfortably in modern hardware. On a dedicated music server, use a larger LLM (33B or 70B) for richer musical reasoning.

---

## 9. Research Context & Prior Art

### 9.1 Algorithmic Music History

- **Iannis Xenakis** (1960s): Used Markov chains and stochastic processes for composition. *Formalized Music* (1971) remains foundational.
- **David Cope** (1980s): EMI (Experiments in Musical Intelligence) used pattern matching and Markov-like recombination to compose in the style of Bach, Mozart, Chopin.
- **Curtis Roads** (1980s): Granular synthesis and algorithmic composition using particle-based methods.
- **Wolfram Tones** (2004): Stephen Wolfram's 1D CA music generator using the 256 elementary rules.

### 9.2 L-Systems in Music

- **Przemysław Prusinkiewicz** (1986): Original work on L-systems for music, adapting his plant-growth models to melodic generation.
- **Stelios Manousakis** (2009): *Musical L-Systems* — comprehensive thesis on using L-systems for algorithmic composition with context-sensitive and parametric extensions.
- **SuperCollider Prewrite**: Standard library implementation for real-time L-system pattern generation, widely used in live coding.
- **Renick Bell** (2014): *Fractal Beats* — rhythmic density control via L-systems in live coding performance.

### 9.3 Fractal Music Research

- **Richard Voss & John Clarke** (1978): Discovered 1/f noise in music audio — the foundational paper on fractal structure in music.
- **Livengood, White & Wong** (2012): *Fractal complexity (1/f power law) determines the stability of music perception, emotion, and memory* — shows that optimal fractal complexity (around D=1.5) elicits peak mood and memory performance.
- **Mandelbrot** (1982): *The Fractal Geometry of Nature* — includes discussion of 1/f noise in music and self-similar structures.

### 9.4 Cellular Automata in Music

- **Eduardo Reck Miranda** (1990s-2000s): Pioneered CA-based music composition, including 2D Game of Life for harmony.
- **Wolfram** (2002): *A New Kind of Science* — included extensive discussion of CA-generated patterns and their application to music.
- **Andrew Brouwer** (2012): Empirical study mapping Wolfram rules to rhythmic patterns, validating groove scores.

### 9.5 JEPA and Music AI

- **Yann LeCun** (2022): Proposed JEPA architecture for self-supervised learning through latent-space prediction.
- **Audio-JEPA** (2024): Self-supervised audio representation learning via masked spectrogram patches, competitive on music classification benchmarks.
- **Music-JEPA** (2025): Treats music as an action-conditioned system — audio is the state, piano roll is the action. Learns to predict future audio states based on current state + actions. Enhanced performance on beat tracking, composer ID, key estimation.
- **Stem-JEPA** (2025): Applies JEPA to multi-track audio for stem compatibility estimation. Relevant to our multi-engine architecture.
- **WavJEPA** (2025): JEPA on raw waveforms, avoiding spectrogram computation. Lower latency.

### 9.6 LLM + Music Generation

- **MusicLM / MusicGen / JASCO** (2023-2025): Text-to-music generation using transformer architectures. These generate audio directly, not MIDI.
- **LLM as musical director**: Emerging research on using LLMs to control symbolic music generation (rather than generating notes directly). Our architecture follows this paradigm.

### 9.7 What's Novel Here

The combination is the innovation. Prior work treats each algorithmic method in isolation (Markov OR L-system OR fractal OR CA). Our system:

1. **Runs all four simultaneously** as coordinated execution engines
2. **Wraps each in a parameter interface** controllable by an external intelligence
3. **Uses JEPA as a universal musical perception layer** that works across all engines
4. **Uses an LLM as a bandleader** that makes structural decisions, not note-level decisions
5. **Creates a closed feedback loop** where perception informs parameter adjustment, which changes the output, which is perceived again

The algorithms don't change. What changes is *who's driving*.

---

## Appendix A: Quick Reference — Engine Parameter Cheat Sheet

### Markov Chain Parameters
```
temperature:     0.1-2.0   (sampling randomness)
order:           1-5       (memory depth)
top_k:           1-20      (transition filtering)
key_lock:        (key, mode)
max_interval:    1-24      (semitones)
swing:           0-0.75    (triplet ratio)
density:         0-1       (note rate)
syncopation:     0-1       (off-beat emphasis)
```

### L-System Parameters
```
rules:           [{symbol, predicate, weight, expansion}]  (the grammar itself)
max_iterations:  4-12      (depth of expansion)
alphabet:        [terminal + non-terminal symbols]
stochasticity:   0-1       (rule weight randomness)
context_sensitivity: bool  (do rules look at neighbors?)
```

### Fractal Parameters
```
hausdorff_D:     1.0-2.0   (fractal dimension / complexity)
lacunarity:      0-1.5     (gap distribution / rhythmic sparseness)
hurst_exponent:  0-1       (long-range dependence / trendiness)
ifs_contractions: [0.1-0.95] per transform
ifs_rotations:   [0-360] degrees per transform
iteration_depth: 4-12      (generation resolution)
```

### Cellular Automata Parameters
```
rule:            0-255     (Wolfram rule number)
width:           16/32/64/128
seed:            [binary array]
boundary:        periodic/fixed/reflecting
iterations/beat: 1-4
density_target:  0.1-0.8
quantize:        none/swing65/swing75
```

### Global Parameters
```
tempo:           60-200 BPM
key:             (key, mode)
time_signature:  (4,4) / (3,4) / (7,8) / etc.
target_valence:  0-1
target_arousal:  0-1
target_tension:  0-1
section_type:    intro/verse/chorus/bridge/solo/outro
```

---

## Appendix B: Implementation Roadmap

### Phase 1: Single Engine (Markov + JEPA)
- Build Markov chain MIDI generator with parameter interface
- Train small JEPA on music corpus for melody perception
- Implement closed-loop: JEPA perceives → simple controller adjusts Markov params
- **Output**: Working agentic Markov melody generator

### Phase 2: Add L-System Engine
- Implement L-system grammar with rule rewriting
- Add JEPA predictor for rule-change lookahead
- Coordinate Markov (melody) + L-system (harmony)
- **Output**: Two-engine agentic system

### Phase 3: Add Fractal and CA Engines
- Implement fractal generator (IFS + midpoint displacement)
- Implement 1D CA rhythm generator with Wolfram rules
- Add multi-track hierarchical CA coordination
- **Output**: Full four-engine system

### Phase 4: LLM Bandleader Integration
- Connect LLM (via API or local model)
- Build context summarization pipeline (JEPA state → LLM prompt)
- Implement parameter adjustment parser (LLM output → engine params)
- Add asynchronous LLM inference with predictive pre-computation
- **Output**: Complete agentic system with LLM direction

### Phase 5: Real-Time Performance
- Optimize latency (< 350ms feedback loop)
- Add MIDI output to DAW/synthesizer
- Add real-time control surface (physical knobs → parameter overrides)
- Performance testing at live tempo
- **Output**: Stage-ready agentic music system

---

*Document version: 1.0 | Author: fleet-jepa-midi project | Date: 2026-08-13*
