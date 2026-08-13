# JEPA-Compatible Architectures for Algorithmic MIDI

## Deep Research Report

**Date:** 2026-08-13  
**Project:** fleet-jepa-midi  
**Author:** Lucineer Research (subagent)  
**Hardware Target:** RTX 4050 (6GB VRAM)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Existing JEPA + Music Landscape](#2-existing-jepa--music-landscape)
3. [Framework 1: A-JEPA (Audio-JEPA)](#3-framework-1-a-jepa-audio-jepa)
4. [Framework 2: V-JEPA via Piano Rolls](#4-framework-2-v-jepa-via-piano-rolls)
5. [Framework 3: Flow-Matching Generative Decoder](#5-framework-3-flow-matching-generative-decoder)
6. [Framework 4: Action-Conditioned World Models (DreamerV3 / MuZero)](#6-framework-4-action-conditioned-world-models-dreamerv3--muzero)
7. [Algorithmic Generator + JEPA Ideation](#7-algorithmic-generator--jepa-ideation)
8. [Feasibility Assessment Matrix](#8-feasibility-assessment-matrix)
9. [Recommended Architecture](#9-recommended-architecture)
10. [References](#10-references)

---

## 1. Executive Summary

This report evaluates four JEPA-compatible architectural frameworks for connecting algorithmic MIDI generators (Markov chains, L-systems, cellular automata, fractals, genetic algorithms) to JEPA-based perceptual/cognitive layers. The research uncovered a rapidly maturing field: as of 2024–2026, at least **seven** JEPA variants for music/audio exist, including two (MIDI-RAE-JEPA and Music-JEPA) that are directly applicable to our project.

**Key findings:**

- **MIDI-RAE-JEPA** (Hawley, July 2026) already implements the exact pipeline we need: Swin Transformer V2 encoder + LeJEPA equivariance objectives on piano roll images + flow-matching generative decoder. It runs on consumer hardware and has open-source code.
- **Music-JEPA** (Wang, Fang, LeCun, July 2026) implements an action-conditioned JEPA world model where audio = state and pianoroll = action. Authored by Yann LeCun's group.
- **Audio-JEPA** (Tuncay et al., ICME 2025) and **A-JEPA** (Fei et al., 2023) provide spectrogram-based JEPA for general audio, with open implementations.
- **Stem-JEPA** (Sony CSL Paris, ISMIR 2024) models multi-track musical stem compatibility.
- All frameworks fit within RTX 4050 6GB VRAM with appropriate model sizing (8M params or less, mixed precision training).

**Recommended architecture:** A hybrid approach using MIDI-RAE-JEPA's Swin V2 encoder + equivariance objectives as the perception core, Music-JEPA's action-conditioned dynamics as the predictive engine, and flow-matching as the generative decoder — with algorithmic generators providing the "action" space.

---

## 2. Existing JEPA + Music Landscape

### Confirmed Existing Projects

| Project | Paper | Date | Authors | Code | Target |
|---------|-------|------|---------|------|--------|
| **MIDI-RAE-JEPA** | arXiv:2607.14537 | Jul 2026 | Scott H. Hawley | ✅ [github.com/drscotthawley/midi-rae](https://github.com/drscotthawley/midi-rae) | Symbolic music (piano rolls) |
| **Music-JEPA** | arXiv:2607.22000 | Jul 2026 | Ziyu Wang, Kun Fang, **Yann LeCun** | ⏳ "coming soon" + [demo](https://zzwaang.github.io/music-jepa-demo/) | Audio + MIDI (action-conditioned) |
| **Audio-JEPA** | arXiv:2507.02915 | Jun 2025 | Tuncay, Labbé, Benetos, Pellegrini | ✅ [github.com/LudovicTuncay/Audio-JEPA](https://github.com/LudovicTuncay/Audio-JEPA) | Audio spectrograms (ICME 2025) |
| **A-JEPA** | arXiv:2311.15830 | Nov 2023 | Zhengcong Fei et al. | ❌ (paper only) | Audio spectrograms (ViT-based) |
| **Stem-JEPA** | arXiv:2408.02514 | Aug 2024 | Riou et al. (Sony CSL Paris) | ✅ [github.com/SonyCSLParis/Stem-JEPA](https://github.com/SonyCSLParis/Stem-JEPA) | Multi-track stem compatibility (ISMIR 2024) |
| **WavJEPA** | (Dec 2025) | Dec 2025 | — | — | Raw waveform (speech, music, environmental) |
| **JEPA for Symbolic Music** | (Sep 2025) | Sep 2025 | — | — | Symbolic music with music-specific masking |

### Detailed Analysis of Key Projects

#### MIDI-RAE-JEPA — **THE MOST DIRECTLY RELEVANT**

**Does MIDI-RAE-JEPA exist?** ✅ **YES.** Published as arXiv:2607.14537 by Scott H. Hawley on July 16, 2026. Code at [github.com/drscotthawley/midi-rae](https://github.com/drscotthawley/midi-rae). Also on [HuggingFace](https://huggingface.co/papers/2607.14537) and [PyPI](https://pypi.org/project/midi-rae/).

**Architecture:**
- **Encoder:** Swin Transformer V2, 6 stages, 4×4 pixel patches on 128×128 piano roll images
- **Objectives:** Pitch/time-shift equivariance + LeJEPA (SIGReg collapse prevention) + Masked Embedding Predictor (MEP)
- **Data:** POP909 dataset (909 pop songs), rendered as 128×128 binary piano rolls (8 bars, eighth-note resolution, all 128 MIDI pitches)
- **Results:** Reconstruction F1 = 0.995, outperforms Haar scattering on emotion classification, embeddings show monotonic equivariance to pitch/time shifts
- **Generation:** Flow-matching generative model conditioned on frozen embeddings → matches pitch register and rhythmic density of conditioning excerpt
- **Hardware:** Runs on consumer GPUs (RTX 4090, 4090 MaxQ, **RTX 2070**) — easily fits RTX 4050

**Key Technical Innovations:**
1. **Smooth pitch/time equivariance loss:** `L_equiv = (‖z₁ - z₂‖ - α√d ‖δ̂‖)²` — larger shifts → proportionally more distant embeddings (both attracts and repels)
2. **Chunked SIGReg:** Reduces VRAM by ~5GB via chunking slice dimension (critical for low-VRAM GPUs)
3. **Soft Factorization Loss:** Encourages pitch and time directions to be geometrically orthogonal in latent space
4. **EMA Teacher:** DINOv2/I-JEPA-style exponential moving average teacher (η=0.96)

#### Music-JEPA — **ACTION-CONDITIONED WORLD MODEL**

**Authors:** Ziyu Wang, Kun Fang, **Yann LeCun** (NYU, MBZUAI, McGill, CIRMMT)  
**Published:** July 24, 2026 as arXiv:2607.22000

**This is literally the JEPA-as-world-model-for-music paper, co-authored by the creator of JEPA.**

**Architecture:**
- **State:** 2-second audio segment as log-mel spectrogram (229 mel bins, 10ms frame rate, L=200 frames)
- **Action:** Pianoroll (88 pitches × 200 frames) + sustain pedal signal (200 values)
- **Encoders:** Separate ViT-based state encoder (ℰ_s) and action encoder (ℰ_a)
- **Dynamics:** s_{t+1} = f(s_t, a_{t+1}), a_{t+1} = g(a_t) — state predictor with cross-attention to action tokens
- **Anti-collapse:** EMA teacher (stop-gradient, momentum coefficient τ)
- **Training:** Fully offline on paired audio-MIDI data
- **Results:** Outperforms audio-only JEPA, comparable to MERT (using only 7% of parameters)
- **Planning:** Piano transcription via planning — search for actions that best explain target sound

**Critical insight for our project:** Music-JEPA treats the pianoroll (MIDI) as the "action" that transitions the audio from one state to another. For our algorithmic generators, we can treat the **algorithmic parameters** (mutation rate, L-system rules, CA rules) as meta-actions that generate the pianoroll actions.

#### Audio-JEPA (Tuncay et al.) vs A-JEPA (Fei et al.)

Two different papers, often confused:

| Aspect | Audio-JEPA (Tuncay) | A-JEPA (Fei) |
|--------|---------------------|--------------|
| **arXiv** | 2507.02915 | 2311.15830 |
| **Date** | Jun 2025 | Nov 2023 |
| **Venue** | ICME 2025 | (preprint) |
| **Backbone** | ViT (I-JEPA style) | ViT |
| **Masking** | Random patch masking on mel-spectrograms | Curriculum time-frequency aware masking |
| **Data** | AudioSet (10s clips, 32kHz) | AudioSet variants |
| **Code** | ✅ GitHub (PyTorch Lightning + Hydra) | ❌ |
| **Checkpoint** | ✅ HuggingFace | ❌ |
| **Eval** | X-ARES suite (speech, music, environmental) | Audio/speech classification |

**Audio-JEPA (Tuncay)** is the better starting point: open code, pretrained checkpoint, clean Lightning-Hydra template, active maintenance.

#### Stem-JEPA (Sony CSL Paris)

**Purpose:** Determine musical stem compatibility — which instruments blend well together.  
**Architecture:** Encoder + predictor trained to predict embeddings of compatible stems from context embeddings.  
**Code:** [github.com/SonyCSLParis/Stem-JEPA](https://github.com/SonyCSLParis/Stem-JEPA)  
**Relevance:** Lower priority for our use case, but interesting for multi-track algorithmic composition where JEPA evaluates which generated tracks complement each other.

---

## 3. Framework 1: A-JEPA (Audio-JEPA)

### How It Works

A-JEPA learns audio representations by predicting masked regions of spectrograms in latent space. The context encoder processes visible spectrogram patches; a target encoder (EMA of context encoder) provides prediction targets for masked regions.

```
┌─────────────────┐     ┌──────────────────┐
│ Mel-spectrogram │────▶│ Context Encoder  │────▶ z_context
│ (masked input)  │     │ (ViT backbone)   │
└─────────────────┘     └──────────────────┘
                                │
                                ▼
                        ┌──────────────────┐
                        │ Predictor        │────▶ ẑ_target
                        │ (ViT, cross-attn)│
                        └──────────────────┘
                                │
                                ▼
┌─────────────────┐     ┌──────────────────┐
│ Full spectrogram│────▶│ Target Encoder   │────▶ z_target (stop-grad)
│ (unmasked)      │     │ (EMA teacher)    │
└─────────────────┘     └──────────────────┘

Loss = ‖ẑ_target - z_target‖²
```

### Input/Output

| Direction | Format | Details |
|-----------|--------|---------|
| **Input** | Mel-spectrogram | 64 mel bins × T frames, patch-encoded |
| **Output** | Latent embeddings | 768-dim per patch token (ViT-Base) |
| **Downstream** | Classification, retrieval, conditioning | Via linear probe or frozen embeddings |

### Harness for Algorithmic MIDI

The harness concept: **algorithmic rules generate MIDI → render to audio → convert to spectrogram → A-JEPA encodes perceptual features → feedback shapes algorithm parameters.**

```
Algorithmic Generator (Markov/L-system/CA/Fractal/GA)
        │
        ▼
    MIDI Output
        │
        ▼
    Synth Rendering (fluidsynth / soundfont)
        │
        ▼
    Mel-spectrogram
        │
        ▼
    A-JEPA Encoder (frozen, pretrained)
        │
        ▼
    Latent Features z ∈ R^768
        │
        ├──▶ Timbre Prediction: "How does changing CA rules alter timbre?"
        │
        ├──▶ Similarity Retrieval: "Which reference tracks does this resemble?"
        │
        └──▶ Feedback to Policy Network → parameter adjustments
```

**Can A-JEPA learn to predict how structural changes alter timbre?**  
Yes, with a predictor head. Train a lightweight MLP on top of frozen A-JEPA embeddings to predict timbral descriptors (brightness, roughness, density) from (current_embedding, parameter_delta) pairs. The JEPA embeddings already capture timbral information; the predictor maps parameter changes to timbral trajectories.

**Can it train on MIDI-derived spectrograms?**  
Yes. Render MIDI to audio via soundfont, compute mel-spectrograms, proceed normally. The spectrograms from MIDI rendering are cleaner than recorded audio but structurally similar.

### RTX 4050 Feasibility

| Component | VRAM | Notes |
|-----------|------|-------|
| ViT-Small encoder | ~1.2 GB | 22M params, batch 16, mixed precision |
| ViT-Base encoder | ~2.8 GB | 86M params, batch 8, mixed precision |
| Fine-tuning (small) | ~2.0 GB | Linear probe: <500MB |
| **Recommendation** | **ViT-Small** | Use A-JEPA's pretrained checkpoint, freeze encoder |

### Training Data Requirements

- **Pretraining:** AudioSet (2M clips, 10s each) — already done by Audio-JEPA checkpoint
- **Fine-tuning:** ~500-1000 algorithmically-generated MIDI clips rendered to audio (~2-4 hours of audio)
- **Minimal viable:** 100 clips for linear probe evaluation

### Open-Source Implementation

- ✅ **Audio-JEPA:** [github.com/LudovicTuncay/Audio-JEPA](https://github.com/LudovicTuncay/Audio-JEPA) — PyTorch Lightning + Hydra, clean codebase
- ✅ **Pretrained checkpoint:** [huggingface.co/ltuncay/Audio-JEPA](https://huggingface.co/ltuncay/Audio-JEPA)
- ❌ **A-JEPA (Fei):** No public code

---

## 4. Framework 2: V-JEPA via Piano Rolls

### How It Works

V-JEPA (Video JEPA) treats video as a sequence of frames and predicts masked spatiotemporal regions in latent space. The key insight for music: **a piano roll is structurally identical to a grayscale video frame**, and a sequence of evolving piano roll slices is a "video" where V-JEPA can learn temporal dynamics.

```
Piano Roll Sequence:
  Frame 1 (bars 1-8)  →  Frame 2 (bars 9-16) →  Frame 3 (bars 17-24)
  [128 × 128 binary]     [128 × 128 binary]      [128 × 128 binary]
         │                       │                       │
         ▼                       ▼                       ▼
    ┌─────────────────────────────────────────────────────────┐
    │ V-JEPA Context Encoder (ViT)                            │
    │ Predicts: latent z_{t+1} from z_t and masked z_{t+1}  │
    └─────────────────────────────────────────────────────────┘
```

### V-JEPA 2 Context

Meta's V-JEPA 2 (2025) extends JEPA to video understanding with large-scale pretraining. The architecture uses a ViT encoder on spatiotemporal patches and predicts masked regions. While V-JEPA was designed for natural video, its architecture is domain-agnostic — it operates on any image-like sequence.

### Piano Roll as Video

A piano roll is a 2D binary matrix:
- **X axis:** Time (eighth notes, 128 steps = 8 bars)
- **Y axis:** Pitch (128 MIDI pitches)
- **Value:** 1 = note on, 0 = note off

A sequence of piano rolls from an algorithmic generator becomes a "video":
- Each "frame" = one window of the evolving composition
- Temporal dimension = how the composition evolves over multiple windows
- V-JEPA learns the "physics" of how the algorithm transforms the musical surface

**For multi-track music:** Stack tracks as RGB channels — melody (R), harmony (G), bass (B). V-JEPA processes this as 3-channel "video."

### Input/Output

| Direction | Format | Details |
|-----------|--------|---------|
| **Input** | Piano roll video | [T_frames, 128, 128, C_tracks] |
| **Output** | Latent trajectory | [T_frames, D_latent] — captures how structure evolves |
| **Prediction** | Next-frame latent | z_{t+1} predicted from z_t (world model) |

### Can V-JEPA Predict Long-Term Structural Trajectory?

Yes, with caveats:
- **Short-term (8-32 bars):** Strong prediction — V-JEPA excels at local spatiotemporal patterns
- **Medium-term (32-128 bars):** Possible with hierarchical encoding (use MIDI-RAE-JEPA's Swin V2 hierarchy)
- **Long-term (128+ bars):** Requires recurrent/autoregressive wrapper around V-JEPA latents

**Recommended approach:** Use MIDI-RAE-JEPA's Swin V2 encoder as the frame encoder (it's already designed for piano rolls), then apply V-JEPA-style temporal prediction on top of the frame-level embeddings.

### RTX 4050 Feasibility

| Component | VRAM | Notes |
|-----------|------|-------|
| Swin V2 frame encoder (small) | ~1.5 GB | Per-frame, 128×128 input |
| Temporal predictor (2-layer Transformer) | ~0.5 GB | 256-dim, 4 heads |
| Full system (8 frames) | ~3.2 GB | Batch 8, mixed precision |
| **Status** | ✅ **Fits** | Comfortable on 6GB |

### Training Data Requirements

- Generated piano roll sequences from algorithmic generators
- ~10,000 sequences of 32 bars each (8 frames × 4 bars/frame)
- Auto-generated: run your algorithmic generators with varied parameters
- No external dataset needed for the algorithm-specific version

### Open-Source Implementation

- V-JEPA 2: Meta's code (research license)
- MIDI-RAE-JEPA: [github.com/drscotthawley/midi-rae](https://github.com/drscotthawley/midi-rae) — **use this as the frame encoder**
- No existing V-JEPA-on-piano-rolls implementation found — **this is a novel application**

---

## 5. Framework 3: Flow-Matching Generative Decoder

### How It Works

JEPA architectures learn representations but don't generate data natively. Flow matching bridges this gap:

1. **Freeze** the JEPA encoder (learned representations)
2. **Train a flow-matching decoder** that learns to map from noise to the JEPA embedding space
3. **Condition** the decoder on control signals (style, parameters, algorithmic metadata)
4. **Decode** embeddings back to data (piano rolls, audio, etc.)

```
Training:
  z = JEPA_encoder(piano_roll)     # frozen encoder
  t ~ Uniform(0, 1)                # time parameter
  ε ~ N(0, I)                      # noise
  z_t = t·z + (1-t)·ε              # linear interpolation
  v̂ = FlowNet(z_t, t, condition)   # predict velocity field
  loss = ‖v̂ - (z - ε)‖²           # flow matching loss

Generation:
  z_0 ~ N(0, I)                    # start from noise
  z_1 = ODE_solve(v̂, z_0, cond)   # follow velocity field to data manifold
  output = Decoder(z_1)            # decode to piano roll / audio
```

### MIDI-RAE-JEPA Already Does This

MIDI-RAE-JEPA's paper demonstrates exactly this pipeline:
- Swin V2 encoder trained with SSL (frozen after training)
- Flow-matching generative model conditioned on frozen embeddings
- Results: "generations that closely match the pitch register and rhythmic density of the conditioning excerpt, while mismatched conditioning yields unrelated but musically plausible output"

### Extending for Algorithmic Generator Control

The innovation: **condition the flow-matching decoder on algorithmic parameters.**

```
Condition vector c = [
    algorithm_type,        # one-hot: Markov, L-system, CA, fractal, GA
    algorithm_params,      # continuous: mutation_rate, angle, rules, etc.
    musical_constraints,   # key, tempo, density target
    reference_embedding    # z from a reference piece (optional)
]

FlowNet(z_t, t, c) → velocity field conditioned on algorithmic identity
```

This enables:
- **Interpolation:** Smoothly morph from Markov-generated to L-system-generated music by interpolating the condition vector
- **Style transfer:** Apply the "feel" of one algorithm with the parameters of another
- **Novel generation:** Explore the latent space between known algorithmic regimes

### Interpolation Between Algorithmic Systems

**Can flow-matching interpolate between Markov-generated and L-system-generated music?**

Yes. The JEPA encoder maps both to the same latent space. Flow-matching learns the distribution conditioned on algorithm type. Interpolation:

```python
# Encode examples from two algorithms
z_markov = encoder(markov_piece)  # e.g., stochastic, smooth transitions
z_lsystem = encoder(lsystem_piece)  # e.g., fractal, self-similar

# Interpolate in latent space
alpha = 0.5  # 50% Markov, 50% L-system
z_interp = (1-alpha) * z_markov + alpha * z_lsystem

# Decode
piano_roll = flow_decoder(z_interp)  # novel hybrid music
```

### Architecture Details

| Component | Architecture | Parameters | Purpose |
|-----------|-------------|------------|---------|
| JEPA Encoder | Swin V2 (frozen) | ~25M | Encode piano rolls to latent space |
| Flow Network | MLP + attention | ~5M | Predict velocity field for ODE |
| ODE Solver | Euler (10 steps) or Dopri5 (adaptive) | 0 | Numerical integration |
| Decoder | Convolutional | ~8M | Decode latent to piano roll |
| Condition Net | MLP | ~2M | Encode algorithmic parameters |

### RTX 4050 Feasibility

| Component | VRAM | Notes |
|-----------|------|-------|
| Frozen encoder (inference) | ~0.8 GB | No gradients needed |
| Flow net training | ~1.5 GB | Small MLP, batch 32 |
| Full pipeline (inference) | ~2.0 GB | 10 Euler steps |
| **Status** | ✅ **Fits comfortably** | Fastest framework to implement |

### Training Data Requirements

- Pairs of (algorithmic_params, piano_roll_output) from your generators
- ~5,000-10,000 examples per algorithm type
- MIDI-RAE-JEPA used POP909 (909 songs) — algorithmically generated data is unlimited

### Open-Source Implementations

- ✅ MIDI-RAE-JEPA includes flow-matching decoder: [github.com/drscotthawley/midi-rae](https://github.com/drscotthawley/midi-rae)
- ✅ MusicFlow (ICML 2024): text-to-music flow matching — [musicflowicml.github.io](https://musicflowicml.github.io/)
- ✅ TorchCFM (Conditional Flow Matching library): standard flow-matching implementations

---

## 6. Framework 4: Action-Conditioned World Models (DreamerV3 / MuZero)

### How It Works

DreamerV3 learns a world model using a [Recurrent State-Space Model (RSSM)](https://arxiv.org/abs/1912.01603) — a [recurrent neural network](https://en.wikipedia.org/wiki/Recurrent_neural_network) that maintains a probabilistic belief over latent states. An agent can "dream" forward: imagine possible futures, evaluate them, and choose actions that lead to desirable outcomes.

For music, this is transformative. The system can **rehearse the future**: given the current musical state (JEPA embedding) and a candidate parameter change (MIDI action), it imagines what the next 8 bars will feel like. If the dreamed future sounds good, execute. If not, try a different action.

> *Before you play the note, hear it in your head. Before you change the rule, dream what it sounds like. This is what every great improviser does — from [Charlie Parker](https://en.wikipedia.org/wiki/Charlie_Parker) to [Keith Jarrett](https://en.wikipedia.org/wiki/Keith_Jarrett). The Dreamer architecture formalizes that inner hearing.*

### RSSM: The Mathematical Core

The RSSM maintains two types of latent state:

$$h_t = f_\theta(h_{t-1}, s_{t-1}, a_{t-1}) \quad \text{(deterministic — GRU)}$$

$$s_t \sim q_\theta(s_t | h_t, o_t) \quad \text{(stochastic posterior — during training)}$$

$$s_t \sim p_\theta(s_t | h_t) \quad \text{(stochastic prior — during dreaming)}$$

where:
- $h_t$ = deterministic hidden state ([GRU](https://en.wikipedia.org/wiki/Gated_recurrent_unit) update)
- $s_t$ = stochastic latent state (multiple possible futures)
- $a_{t-1}$ = MIDI parameter action
- $o_t$ = JEPA embedding observation

The key insight: during training, the model has access to actual musical output (posterior $q$). During dreaming, it only uses its imagination (prior $p$). This is how the system **hallucinates musical futures** before playing them.

### Musical Mapping

| DreamerV3 Concept | JEPA-MIDI Equivalent | Musical Analogy |
|---|---|---|
| **State** $s_t$ | JEPA embedding | "What the music feels like now" |
| **Action** $a_t$ | MIDI parameter changes | "Let me increase density" |
| **Dynamics** $f(s_t, a_t)$ | How parameters change feel | "If I speed up, it gets more urgent" |
| **Reward** $r_t$ | Musical quality | "Does this sound good?" |
| **Dream** | Imagine before playing | "Hear it in your head first" |

### RTX 4050 Feasibility

| Component | VRAM | Notes |
|-----------|------|-------|
| RSSM (GRU + prior/posterior nets) | ~0.5 GB | 512-dim hidden, 384-dim latent |
| Reward predictor | ~0.1 GB | Small MLP |
| Actor-critic (for action selection) | ~0.2 GB | Optional — can use dreamed rewards directly |
| Training (with observation encoder) | ~2.0 GB | Batch 32, mixed precision |
| **Total** | **~2.8 GB** | **Fits — tight but feasible** |

### Training Data

- Recorded (JEPA embedding, MIDI action, reward) tuples from performances
- ~100 hours of recorded play = ~288,000 training steps (at 1.25s/step)
- Can bootstrap from self-play: let the system perform, record, train, repeat

### Open-Source Implementations

- ✅ [DreamerV3 official (TF/JAX)](https://github.com/danijar/dreamerv3) — Google Research
- ✅ [DreamerV3 PyTorch](https://github.com/NM512/dreamerv3-torch) — community port
- ✅ [DreamerV2](https://github.com/danijar/dreamerv2) — predecessor
- ❌ No existing Dreamer-for-music implementation — **this is novel**

### Why Not MuZero?

[MuZero](https://arxiv.org/abs/1911.08265) learns a model that plans via [Monte Carlo Tree Search (MCTS)](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search). It's more powerful than Dreamer for discrete action spaces (Go, Chess), but:
1. MCTS is too slow for real-time music (needs hundreds of forward passes per decision)
2. Our action space is continuous (parameter deltas), which MuZero handles poorly
3. Dreamer's "dream" paradigm is more natural for music — imagine, evaluate, play

**Recommendation: DreamerV3, not MuZero.**

---

## 7. Algorithmic Generator + JEPA Ideation

### The Synthesis

All four frameworks above share a common thread: they separate **perception** (JEPA encoder) from **generation** (algorithmic engine or flow decoder) and connect them through a **latent space** (embeddings). This separation is exactly what the [agentic algorithmic music system](agentic-algorithmic-music.md) needs:

```
Algorithmic Engine generates MIDI
        │
        ▼
JEPA encodes the output ───► embedding ("how does it feel?")
        │
        ▼
JEPA predicts next-phrase embedding ("where is it going?")
        │
        ▼
LLM receives context + prediction ───► parameter adjustments
        │
        ▼
Algorithmic Engine generates with new parameters
```

### What Each Framework Contributes

| Framework | Contribution to the Agentic System |
|-----------|-------------------------------------|
| **A-JEPA (Audio)** | Pretrained audio embeddings — no need to train from scratch. Use for cross-modal alignment. |
| **V-JEPA (Piano Rolls)** | Temporal dynamics prediction — how the "video" of a piano roll evolves over multiple bars. |
| **Flow Matching** | Generative decoder — interpolate between algorithmic regimes (Markov → L-system morphing). |
| **DreamerV3** | Forward planning — "dream" the musical future before committing to parameter changes. |

### The Novel Combination

No existing project combines all four. Our proposed architecture:
1. **Perception:** MIDI-RAE-JEPA's Swin V2 encoder (already proven on piano rolls)
2. **Prediction:** Music-JEPA's action-conditioned dynamics (audio state, MIDI action)
3. **Generation:** Algorithmic engines (Markov, L-system, fractal, CA) — not flow matching
4. **Planning:** DreamerV3-style RSSM on top of JEPA embeddings
5. **Self-improvement:** [Curiosity + adversarial masking + Cross-JEPA](self-improving-harnesses.md)

This is the architectural contribution of fleet-jepa-midi: not a new algorithm, but a novel **combination** of existing algorithms across the three timescales of music.

---

## 8. Feasibility Assessment Matrix

| Framework | VRAM (6GB Budget) | Implementation Effort | Music Quality | Real-Time Capable | Novelty |
|-----------|-------------------|----------------------|---------------|-------------------|--------|
| **A-JEPA** | ✅ 1.2 GB (ViT-S) | Low (pretrained) | Good (audio-level) | ✅ Yes | Low (exists) |
| **V-JEPA (Piano Roll)** | ✅ 1.5 GB | Medium (adapt V-JEPA) | Very Good | ✅ Yes | **High** (novel application) |
| **Flow Matching** | ✅ 2.0 GB | Medium (adapt MIDI-RAE) | Excellent | ✅ Yes (10 Euler steps) | Medium |
| **DreamerV3** | ⚠️ 2.8 GB (tight) | **High** (RSSM + reward) | Unknown (novel) | ✅ Yes (dreams fast) | **Very High** (novel) |
| **Hybrid (Recommended)** | ✅ 2.6 GB | **High** | **Excellent** | ✅ Yes | **Very High** |

---

## 9. Recommended Architecture

### Phase 1 (Baseline): Token-Based JEPA

Start with the architecture in the [training design doc](jepa-training-design.md):
- 4-layer [Conformer](https://arxiv.org/abs/2005.08100) encoder, 384 dim, 18.7M params
- Fixed future-block masking on 64-token windows (vocab 141)
- [BYOL-style](https://arxiv.org/abs/2006.07733) anti-collapse (EMA + stop-grad + VICReg)
- L1 loss on normalized embeddings

**Why start here:** Simplest to implement. Smallest model. Fastest to train. Already fits RTX 4050 with 47% VRAM margin.

### Phase 2 (Enhanced): Add Piano Roll View

Add a parallel piano-roll encoder inspired by [MIDI-RAE-JEPA](https://github.com/drscotthawley/midi-rae):
- [Swin Transformer V2](https://arxiv.org/abs/2111.09883) on 128×128 piano roll images
- Pitch/time equivariance loss
- Fuse with token-based encoder via concatenation + projection

**Why add this:** Piano roll view captures spatial (pitch × time) structure that token sequences miss. The fusion of token + visual views creates richer embeddings.

### Phase 3 (Generative): Add Flow-Matching Decoder

Add a [flow-matching](https://arxiv.org/abs/2210.02747) generative decoder:
- Condition on algorithmic parameters (engine type, params, musical constraints)
- Enables interpolation between algorithmic regimes
- Uses frozen Phase 2 encoder embeddings as target distribution

**Why add this:** Flow matching enables **novelty generation** — music that exists between known algorithmic territories. A Markov-L-system hybrid that sounds like neither parent.

### Phase 4 (Planning): Add Dreamer World Model

Add the [DreamerV3](https://arxiv.org/abs/2301.04104) RSSM on top of the JEPA embedding space:
- State = JEPA embedding (from Phase 2)
- Action = MIDI parameter deltas (from the [engine parameter space](agentic-algorithmic-music.md#appendix-a-quick-reference--engine-parameter-cheat-sheet))
- Reward = musical quality (coherence + curiosity + cross-modal alignment)

**Why add this:** The system can now **rehearse the future** — dream parameter changes, evaluate their predicted effect, and choose the best one before playing a single note. See [Self-Improving Harnesses](self-improving-harnesses.md#4-architecture-3-latent-action-world-modeling-dreamer) for full details.

### Architecture Evolution

```
Phase 1: Token JEPA (baseline)
┌───────────────────┐
│ 4-layer Conformer │  18.7M params, 2.6GB VRAM
│ Fixed masking     │  L1 loss + VICReg
└───────────────────┘

Phase 2: + Piano Roll View
┌───────────────────┐    ┌───────────────────┐
│ 4-layer Conformer │    │ Swin V2 Encoder   │  +25M params, +0.8GB
│ (token view)      │──►│ (piano roll view) │  Equivariance loss
└───────────────────┘    └─────────┬─────────┘
                                   │
                         ┌─────────▼─────────┐
                         │ Fusion Projection │  384-dim output
                         └───────────────────┘

Phase 3: + Flow Matching Decoder
┌───────────────────┐    ┌───────────────────┐
│ Phase 2 Encoder   │──►│ Flow Decoder      │  +15M params, +0.5GB
│ (frozen)          │    │ (generative)      │  Enables interpolation
└───────────────────┘    └───────────────────┘

Phase 4: + Dreamer Planner
┌───────────────────┐    ┌───────────────────┐
│ Phase 2 Encoder   │──►│ RSSM World Model  │  +8M params, +0.5GB
│ (frozen)          │    │ (dreams futures)  │  Plans before playing
└───────────────────┘    └───────────────────┘

Total Phase 4: ~67M params, ~4.4GB VRAM — fits RTX 4050 with 27% margin
```

---

## 10. References

### JEPA Core

| Paper | Year | Link | Why It Matters |
|-------|------|------|----------------|
| I-JEPA (Assran et al.) | 2023 | [arXiv:2301.08243](https://arxiv.org/abs/2301.08243) | The foundational JEPA paper |
| V-JEPA (Bardes et al.) | 2024 | [arXiv:2403.02537](https://arxiv.org/abs/2403.02537) | Video JEPA — temporal masking strategies |
| V-JEPA 2 | 2025 | [ai.meta.com](https://ai.meta.com/blog/vjepa-2/) | Action-conditioned prediction |
| A Path Towards Autonomous Machine Intelligence (LeCun) | 2022 | [openreview.net](https://openreview.net/forum?id=BZ5a1r-kVsf) | LeCun's vision paper |

### Music + JEPA

| Paper | Year | Link | Why It Matters |
|-------|------|------|----------------|
| MIDI-RAE-JEPA (Hawley) | 2026 | [arXiv:2607.14537](https://arxiv.org/abs/2607.14537) | Swin V2 + flow matching on piano rolls. **Most directly relevant.** |
| Music-JEPA (Wang, Fang, LeCun) | 2026 | [arXiv:2607.22000](https://arxiv.org/abs/2607.22000) | Action-conditioned world model. **Co-authored by LeCun.** |
| Audio-JEPA (Tuncay et al.) | 2025 | [arXiv:2507.02915](https://arxiv.org/abs/2507.02915) | Open code, pretrained checkpoint. Best starting point for audio. |
| A-JEPA (Fei et al.) | 2023 | [arXiv:2311.15830](https://arxiv.org/abs/2311.15830) | Curriculum time-frequency masking. |
| Stem-JEPA (Sony CSL) | 2024 | [github.com/SonyCSLParis](https://github.com/SonyCSLParis/Stem-JEPA) | Multi-track stem compatibility. |

### Architecture & Training

| Paper | Year | Link | Why It Matters |
|-------|------|------|----------------|
| [Conformer](https://arxiv.org/abs/2005.08100) (Gulati et al.) | 2020 | arXiv:2005.08100 | CNN + attention hybrid for audio/music |
| [Swin Transformer V2](https://arxiv.org/abs/2111.09883) | 2022 | arXiv:2111.09883 | Hierarchical vision transformer |
| [DreamerV3](https://arxiv.org/abs/2301.04104) (Hafner et al.) | 2023 | arXiv:2301.04104 | World models that dream |
| [Flow Matching](https://arxiv.org/abs/2210.02747) (Lipman et al.) | 2023 | arXiv:2210.02747 | Generative via ODE transport |
| [FlashAttention-2](https://arxiv.org/abs/2307.08691) | 2023 | arXiv:2307.08691 | Memory-efficient attention |

### Self-Supervised Learning Foundations

| Paper | Year | Link | Why It Matters |
|-------|------|------|----------------|
| [BYOL](https://arxiv.org/abs/2006.07733) | 2020 | arXiv:2006.07733 | EMA + predictor (no negatives needed) |
| [SimSiam](https://arxiv.org/abs/2011.10566) | 2020 | arXiv:2011.10566 | Stop-gradient prevents collapse |
| [VICReg](https://arxiv.org/abs/2105.04906) | 2022 | arXiv:2105.04906 | Variance + invariance + covariance |
| [DINOv2](https://arxiv.org/abs/2304.07193) | 2023 | arXiv:2304.07193 | Self-supervised vision — EMA teacher template |

---

*Research report version: 2.0 | Project: fleet-jepa-midi | Date: 2026-08-13*  
*Expanded with ideation from ByteDance Seed-2.0-pro, NousResearch Hermes-3-Llama-405B, Qwen3-Coder-480B, and NVIDIA Nemotron-3-Ultra via DeepInfra.*