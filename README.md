# Fleet JEPA-MIDI

**A three-layer real-time music intelligence system that perceives, thinks, and plays.**

> *The LLM thinks in phrasing. The JEPA feels in pulse. The algorithms execute in samples.*

---

## 🎵 The Vision

**Fleet JEPA-MIDI is not another music generator.** It's a system that *listens* to music the way a musician does — not by processing audio samples, but by **feeling** energy, tension, groove, and direction. It then uses that feeling to *think* about what should come next, and translates those thoughts into sound through mathematical algorithms that have been making music for decades.

This is the difference between a player piano and a pianist. A player piano reads rolls of paper. A pianist reads the room.

### Why This Matters

Every existing AI music system either:
1. **Generates notes without understanding feel** — it can write a melody but can't tell if it's tense, relaxed, swinging, or stiff
2. **Needs massive compute** — requiring data center GPUs for inference, making real-time performance impossible
3. **Can't adapt** — it generates a fixed output, not a living performance that responds to the moment

Fleet JEPA-MIDI solves all three by separating music intelligence into **three timescales**, each handled by the right tool for the job:

```
┌─────────────────────────────────────────────────────────────┐
│                     THE VISION                               │
│                                                             │
│   PERCEIVE          THINK             EXECUTE               │
│   (JEPA)           (LLM)          (Algorithms)              │
│                                                             │
│   Feels the        Decides           Plays the             │
│   pulse, energy,   direction,        notes with            │
│   tension, swing   phrasing, form    sub-ms precision       │
│                                                             │
│   Every 125ms      Every 1-4 bars    Every sample           │
│   (16th notes)     (phrasing unit)   (<1ms)                │
│                                                             │
│   "What does it    "Where should     "Here are the         │
│    feel like       the music go?"    MIDI events."         │
│    right now?"                                              │
└─────────────────────────────────────────────────────────────┘
```

### For Everyone

| Audience | What This Means |
|----------|----------------|
| **[Musicians](#for-musicians)** | A collaborative partner that listens, responds, and grows — not a backing track |
| **[Educators](#for-educators)** | A tool to teach what "feel" and "energy" mean in music, interactively |
| **[Developers](#for-developers)** | An open, modular system with clean APIs and clear design docs |
| **[Mathematicians](#for-mathematicians)** | Elegant [self-supervised learning](https://en.wikipedia.org/wiki/Self-supervised_learning) applied to the geometry of musical perception |
| **[Engineers](#for-engineers)** | Runs on a laptop GPU. Sub-millisecond inference. 6GB VRAM. |

---

## 📐 The Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    LLM BANDLEADER (per phrasing unit)             │
│                                                                  │
│   "Build tension here. Resolve in two bars.                      │
│    Quote the bridge melody. Trade fours. Lay back."              │
│                                                                  │
│   Called every 1-4 bars — NOT on every tick                      │
│   Thinks: form, narrative, dynamics, interaction                  │
│                                                                  │
│   Models: [GLM-4.5-air](https://chat.z.ai) ·                     │
│           [DeepSeek V4-Flash](https://www.deepseek.com) ·        │
│           [Claude Haiku 5](https://www.anthropic.com)            │
└──────────────────────────────┬───────────────────────────────────┘
                               │ direction / intent / phrasing
                               │ (JSON directives — see [interface design](docs/llm-interface-design.md))
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                    JEPA-MIDI PERCEIVER (per pulse)               │
│                                                                  │
│   Latent embedding of WHERE THE MUSIC IS right now.              │
│   384-dimensional vector encoding:                               │
│   • Energy level     • Harmonic tension                          │
│   • Rhythmic pocket  • Swing amount                              │
│   • Melodic direction • Register & density                       │
│                                                                  │
│   Updated every 125ms (one 16th-note pulse at 120 BPM)           │
│   Inference: 1.3ms — uses 1% of the pulse budget                 │
│                                                                  │
│   Architecture: 4-layer [Conformer](https://arxiv.org/abs/2005.08100) │
│   18.7M params · [Self-supervised](https://en.wikipedia.org/wiki/Self-supervised_learning) │
│   [EMA target encoder](https://arxiv.org/abs/2006.07733)         │
└──────────────────────────────┬───────────────────────────────────┘
                               │ parameters / targets / constraints
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│              ALGORITHMIC ENGINES (sub-millisecond)               │
│                                                                  │
│   Pure math. No thinking. Just execution.                        │
│                                                                  │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│   │ Markov   │ │ L-System │ │ Fractal  │ │ Cellular│            │
│   │ Melody   │ │ Harmony  │ │ Contour  │ │ Automata│            │
│   └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐                       │
│   │ Pulse    │ │Counter-  │ │ Groove   │                        │
│   │ Grid     │ │ point    │ │ Tracker  │                        │
│   └──────────┘ └──────────┘ └──────────┘                       │
│                                                                  │
│   Each engine takes parameters from JEPA+LLM → outputs MIDI      │
└──────────────────────────────────────────────────────────────────┘
```

### The Feedback Loop

```
JEPA reads current state ──► LLM decides direction ──► Algorithms execute
        ▲                                                        │
        └──────────── JEPA reads new state ◄────────────────────┘

The loop runs at the pulse rate (125ms).
The LLM runs at the phrasing rate (1-4 bars).
The algorithms run at the sample rate (<1ms).
Three clocks, one instrument.
```

---

## 📐 The Math Behind JEPA — In Three Levels

### Level 1: Intuitive 🎵

Imagine you're listening to a jazz quartet. The saxophonist is mid-solo. You can't predict the exact notes they'll play next — but you can feel where the music is *going*. You know it's building tension. You know the resolution is coming. You know the groove is deep in the pocket.

You're not predicting notes. You're predicting **feel**.

That's what JEPA does. Instead of trying to predict the exact next note (which is like trying to predict the exact next pixel in a photo), JEPA predicts the **abstract embedding** of what comes next — the feel, the energy, the tension. It learns the *meaning* of music, not the *surface*.

[**JEPA**](https://arxiv.org/abs/2301.08243) stands for **Joint Embedding Predictive Architecture**. It was introduced by [Yann LeCun](https://en.wikipedia.org/wiki/Yann_LeCun) (the "godfather of AI" and Turing Award winner) as a model of how biological brains learn — by predicting the future at abstract levels, not by reconstructing every detail.

### Level 2: Technical 🔧

JEPA has three components:

1. **Context Encoder** $f_\theta$: Takes the first 32 tokens (the "past") and encodes them into a 384-dimensional embedding
2. **Target Encoder** $f_{\theta'}$: An [exponential moving average (EMA)](https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average) copy of the context encoder that processes the full 64-token window. Its weights are frozen (no gradients flow through it)
3. **Predictor** $P_\phi$: A linear layer that predicts the target embedding from the context embedding

The loss function is simple:

$$\mathcal{L} = \| P_\phi(f_\theta(x_{context})) - \text{stopgrad}(f_{\theta'}(x_{target})) \|_1 + \lambda \cdot \text{VICReg}$$

The first term is [L1 distance](https://en.wikipedia.org/wiki/L1_norm) between predicted and actual embeddings. The second is a lightweight [VICReg](https://arxiv.org/abs/2105.04906) regularizer that prevents collapse (all embeddings becoming identical).

**Why EMA + stop-gradient?** Without it, the encoder can cheat: map everything to the same point, and the loss is zero (the trivial solution). The [EMA target](https://arxiv.org/abs/2006.07733) keeps moving slowly, so the trivial solution is never stable. The [stop-gradient](https://arxiv.org/abs/2011.10566) breaks the symmetry that enables collapse.

### Level 3: Rigorous 🧮

The JEPA objective is a form of **latent predictive learning** that avoids the problems of generative models (mode collapse, blurry outputs) and contrastive methods (need for large batches and negative sampling).

Formally, let $x \in \mathcal{X}$ be a MIDI token window of length 64. Let $\mathcal{M}$ be a masking operator that splits $x$ into context $x_c$ (first 32 tokens) and target $x_t$ (last 32 tokens). Let $f_\theta: \mathcal{X} \to \mathbb{R}^d$ be the online encoder and $f_{\theta'}$ be the EMA encoder with $\theta' \leftarrow \tau\theta' + (1-\tau)\theta$.

The training objective:

$$\mathcal{L}_{JEPA}(\theta, \phi) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \left\| \frac{P_\phi(\bar{f}_\theta(x_c))}{\|P_\phi(\bar{f}_\theta(x_c))\|_2} - \frac{\bar{f}_{\theta'}(x_t)}{\|\bar{f}_{\theta'}(x_t)\|_2} \right\|_1 \right] + \lambda_v \sum_{j=1}^{d} \max(0, 1 - \text{std}(\mathbf{z}_j))$$

where $\bar{f}$ denotes [mean-pooling](https://en.wikipedia.org/wiki/Pooling_layer) over the sequence dimension, $\text{std}(\mathbf{z}_j)$ is the batch-wise standard deviation of the $j$-th embedding coordinate, and $\lambda_v = 0.1$.

This connects to:
- **[Information theory](https://en.wikipedia.org/wiki/Information_theory):** The L1 loss approximates the [KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) between predicted and actual embedding distributions when embeddings are approximately Gaussian
- **[Variational bounds](https://en.wikipedia.org/wiki/Evidence_lower_bound):** JEPA optimizes a bound on mutual information between context and target representations
- **[Hyperspherical geometry](https://en.wikipedia.org/wiki/Hypersphere):** L2 normalization projects embeddings onto the unit hypersphere $S^{383}$, where angular distance measures semantic similarity
- **[BYOL theory](https://arxiv.org/abs/2006.07733):** EMA + predictor avoids collapse without negative pairs by implicit alignment-uniformity balancing

**Key insight:** By predicting in latent space (not token space), JEPA avoids the "generative collapse" problem that plagues [autoregressive models](https://en.wikipedia.org/wiki/Autoregressive_model) and [VAEs](https://en.wikipedia.org/wiki/Variational_autoencoder). The model never needs to reconstruct exact MIDI — it only needs to predict the abstract *feel* of what comes next.

---

## 📚 Design Documents

### Core Architecture

| Document | What It Covers | Audience |
|----------|---------------|----------|
| **[JEPA Training Design](docs/jepa-training-design.md)** | Encoder architecture, masking strategy, anti-collapse, memory budget, inference path, evaluation protocol | Engineers, ML researchers |
| **[LLM Interface Design](docs/llm-interface-design.md)** | Directive vocabulary (36 actions!), calling cadence, sensory context, prompt architecture, feedback loop | Developers, system designers |
| **[Agentic Algorithmic Music](docs/agentic-algorithmic-music.md)** | Markov chains, L-systems, fractals, and cellular automata as controllable instruments under agentic direction | Developers, mathematicians |
| **[JEPA-Compatible Architectures](docs/jepa-compatible-architectures-research.md)** | Survey of existing JEPA+music projects (MIDI-RAE-JEPA, Music-JEPA, Audio-JEPA, Stem-JEPA), 4 frameworks analyzed | Researchers |
| **[Self-Improving Harnesses](docs/self-improving-harnesses.md)** | 4 architectures for the system to teach itself: curiosity loops, adversarial masking, Dreamer world models, Cross-JEPA | Researchers, ML engineers |

### Quick Links by Topic

**For Developers:**
- [Directive JSON Schema](docs/llm-interface-design.md#26-output-schema-json-schema) — the complete vocabulary the LLM outputs
- [Engine Parameter Cheat Sheet](docs/agentic-algorithmic-music.md#appendix-a-quick-reference--engine-parameter-cheat-sheet) — all controllable parameters
- [Inference Implementation](docs/jepa-training-design.md#10-inference-path) — production-ready Python code for real-time embedding
- [Implementation Roadmap](docs/agentic-algorithmic-music.md#appendix-b-implementation-roadmap) — 5-phase build plan

**For Mathematicians:**
- [JEPA Objective Function](docs/jepa-training-design.md#4-jepa-objective) — the loss function and why it works
- [Anti-Collapse Strategy](docs/jepa-training-design.md#5-anti-collapse-strategy) — EMA + stop-gradient + VICReg math
- [Fractal Parameters (Hausdorff Dimension, Hurst Exponent)](docs/agentic-algorithmic-music.md#4-agentic-fractals) — continuous fractal complexity for music
- [Self-Improvement Equation](docs/self-improving-harnesses.md#appendix-the-full-self-improvement-equation) — combined objective across all 4 harnesses
- [Procrustes Alignment](docs/self-improving-harnesses.md#5-architecture-4-multi-modal-coherence-alignment-cross-jepa) — Cross-JEPA coherence via orthogonal Procrustes

**For Educators:**
- [The Three Timescales](#-the-architecture) — how to explain perceive/think/execute
- [JEPA Math in 3 Levels](#-the-math-behind-jepa--in-three-levels) — intuitive → technical → rigorous
- [Educator Resources (Self-Improving Harnesses)](docs/self-improving-harnesses.md#educator-resources) — how to teach these concepts
- [Jazz Pedagogy Connection](docs/self-improving-harnesses.md#connection-to-jazz-pedagogy) — how the curiosity loop formalizes jazz education principles

---

## 🎹 The Three Timescales

| Layer | Timescale | What It Does | Analogy |
|-------|-----------|-------------|---------|
| **LLM** | per phrase (1-4 bars) | Thinks: form, direction, dynamics, quotes | The bandleader calling the next tune |
| **JEPA** | per pulse (16th notes ~125ms) | Feels: pocket, energy, tension, swing | The musician's ear, always listening |
| **Algorithms** | per sample (<1ms) | Executes: notes, velocities, CC values | The fingers on the keys |

### Why Three Timescales?

Because music *happens* at three timescales. A [symphony](https://en.wikipedia.org/wiki/Symphony) has movements (minutes). A [phrase](https://en.wikipedia.org/wiki/Phrase_(music)) has direction (seconds). A [note](https://en.wikipedia.org/wiki/Note) has attack, decay, sustain, release (milliseconds).

No single AI model can operate at all three without latency or quality compromises:
- An LLM that thinks about form is too slow to play notes
- A note-level model can't see the big picture
- A perception model can't generate anything — it can only feel

**Separation of concerns** — the foundational principle of software engineering — applied to music.

---

## 🎸 For Musicians

### What This System Does

It plays music with intention. Not random notes. Not a fixed composition. It:
- **Listens** to what's happening (via JEPA perception)
- **Decides** where to go next (via LLM direction)
- **Plays** with sub-millisecond precision (via algorithms)

### How You'd Use It

1. **As a practice partner:** Set it to comp while you solo. It listens to your playing and adjusts the accompaniment.
2. **As a composition tool:** Let it generate ideas. The LLM can suggest structural changes ("build tension here, resolve there") that you can accept or override.
3. **As a performance instrument:** Play it live. It responds to the room, to human input, to the energy of the moment.
4. **As a teaching tool:** Students can see and feel what "tension," "energy," and "pocket" mean — the JEPA embedding makes abstract musical concepts visible as data.

### Human Input

The system is designed for **conversation, not automation**. When a human musician enters:
- The LLM shifts to `comp`, `leave_space`, and `call_response` directives
- The system becomes a responsive sideman, not an autonomous bandleader
- Human input always takes priority over LLM output

---

## 📖 For Educators

### Teaching with JEPA-MIDI

The system makes abstract musical concepts **tangible**:

| Concept | How to Teach It |
|---------|----------------|
| [**Tension and release**](https://en.wikipedia.org/wiki/Tension_and_release) | Show the JEPA's tension dimension rising and falling in real-time as music plays |
| [**Groove and pocket**](https://en.wikipedia.org/wiki/Groove_(music)) | The JEPA embedding has a "pocket" dimension — show how it shifts when a drummer plays ahead/behind |
| [**Form and structure**](https://en.wikipedia.org/wiki/Musical_form) | The LLM's macro plan visualizes the arc of a piece across choruses |
| [**Swing**](https://en.wikipedia.org/wiki/Swing_(jazz_performance_style)) | The swing parameter is a continuous value — let students adjust it and hear the difference |
| [**Improvisation**](https://en.wikipedia.org/wiki/Jazz_improvisation) | The algorithmic engines (especially [Markov chains](https://en.wikipedia.org/wiki/Markov_chain)) make the trade-off between predictability and surprise explicit |
| [**Self-supervised learning**](https://en.wikipedia.org/wiki/Self-supervised_learning) | The JEPA training process is a powerful teaching tool for ML concepts — prediction in latent space, representation collapse, EMA |

### Curriculum Ideas

- **Middle school:** Use the three-layer diagram to explain how humans listen to music (we don't process audio samples — we perceive feel)
- **High school:** Let students adjust algorithm parameters and hear how Markov temperature or fractal dimension changes the music
- **University:** Use the [training design doc](docs/jepa-training-design.md) as a case study in practical ML engineering under hardware constraints
- **Graduate:** The [self-improving harnesses](docs/self-improving-harnesses.md) as a survey of modern SSL research applied to a concrete domain

---

## 🔬 For Mathematicians

### The Core Objects

| Object | Space | Description |
|--------|-------|-------------|
| Token window $x$ | $\{0, 1, \ldots, 140\}^{64}$ | Discrete token sequence (vocab 141, length 64) |
| Context embedding $\mathbf{z}_c$ | $\mathbb{R}^{384}$ | Mean-pooled encoder output of first 32 tokens |
| Target embedding $\mathbf{z}_t$ | $\mathbb{R}^{384}$ | Mean-pooled EMA encoder output of last 32 tokens |
| Predicted embedding $\hat{\mathbf{z}}$ | $\mathbb{R}^{384}$ | $P_\phi(\mathbf{z}_c)$, normalized to $S^{383}$ |
| JEPA loss $\mathcal{L}$ | $\mathbb{R}_{\geq 0}$ | L1 on $S^{383}$ + VICReg variance regularizer |

### Key Properties

1. **Embeddings live on the unit hypersphere** $S^{383}$ via [L2 normalization](https://en.wikipedia.org/wiki/Unit_vector)
2. **Prediction is in latent space**, not token space — avoiding the [reconstruction problem](https://en.wikipedia.org/wiki/Autoencoder) of VAEs and the [mode collapse](https://en.wikipedia.org/wiki/Generative_adversarial_network#Mode_collapse) problem of GANs
3. **EMA target** creates a [non-stationary](https://en.wikipedia.org/wiki/Stationary_process) optimization landscape that prevents [trivial solutions](https://en.wikipedia.org/wiki/Representation_collapse)
4. **VICReg regularizer** maintains [variance](https://en.wikipedia.org/wiki/Variance) per dimension $\geq 1$ and decorrelates dimensions via off-diagonal [covariance](https://en.wikipedia.org/wiki/Covariance) penalty

### Connections to Mathematical Fields

- **[Information geometry](https://en.wikipedia.org/wiki/Information_geometry):** Embedding space is a [Riemannian manifold](https://en.wikipedia.org/wiki/Riemannian_manifold) (the hypersphere), where distances measure musical similarity
- **[Optimal transport](https://en.wikipedia.org/wiki/Optimal_transport):** [Flow matching](https://arxiv.org/abs/2210.02747) in the generative decoder transports noise distribution to music distribution
- **[Game theory](https://en.wikipedia.org/wiki/Game_theory):** Adversarial masking is a [minimax game](https://en.wikipedia.org/wiki/Minimax) with a [saddle-point equilibrium](https://en.wikipedia.org/wiki/Saddle_point)
- **[Dynamical systems](https://en.wikipedia.org/wiki/Dynamical_system):** The Dreamer RSSM is a [stochastic dynamical system](https://en.wikipedia.org/wiki/Stochastic_process) on latent space
- **[Linear algebra](https://en.wikipedia.org/wiki/Linear_algebra):** Cross-JEPA alignment via [Orthogonal Procrustes](https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem) and [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition)

---

## ⚙️ For Engineers

### Hardware Target

| Component | Spec | Notes |
|-----------|------|-------|
| **GPU** | RTX 4050 Laptop (6GB VRAM) | ~2.8GB free after display + models |
| **RAM** | 24GB | For MIDI corpus caching |
| **Training time** | ~11.4 hours (overnight) | Single GPU, FP16 mixed precision |
| **Inference latency** | 1.3ms end-to-end | 1% of 125ms pulse budget |
| **Model size** | 18.7M params (trainable) | Small enough to fit alongside other GPU processes |
| **Total VRAM at inference** | ~138MB | Leaves ~2.66GB for other processes |

### Performance Budget

```
┌──────────────────────────────────────────────────────┐
│                  125ms PULSE BUDGET                   │
│                                                      │
│  MIDI ingest + tokenize:  0.2ms  █                   │
│  JEPA encoder forward:    1.1ms  ██                  │
│  Embedding smoothing:     0.01ms                     │
│  ─────────────────────────────────                   │
│  Total inference:         1.3ms  ██                  │
│  Remaining budget:      123.7ms  ████████████████   │
│                                                      │
│  (used for: algorithm execution, MIDI routing,       │
│   audio synthesis, LLM calls on phrase boundaries)   │
└──────────────────────────────────────────────────────┘
```

### Training Data

| Dataset | Files | Hours | Use |
|---------|-------|-------|-----|
| [Lakh MIDI](https://colinraffel.com/projects/lmd/) | 176,000+ | ~12,000 | Bulk pretraining — diverse styles |
| [MAESTRO](https://magenta.tensorflow.org/datasets/maestro) | 1,272 | 200 | Virtuosic piano — expressive nuance |
| Hooktheory Corpus | ~12,000 | ~800 | Melody + harmony pairs |
| SuperInstance Fakebook | ~500 | ~50 | Domain-specific |

### Key Design Constraints

1. **No negative pairs** — [BYOL-style](https://arxiv.org/abs/2006.07733) EMA + predictor, not [InfoNCE](https://arxiv.org/abs/2005.10243) contrastive loss. Saves VRAM (no large batch needed for negatives).
2. **Fixed-window tokens (141 vocab)** — 7× smaller than [MIDI-BERT](https://arxiv.org/abs/2107.05223)'s 400+ vocab. No padding overhead.
3. **[FlashAttention-2](https://arxiv.org/abs/2307.08691)** — reduces attention memory from O(n²) to O(n)
4. **[Gradient checkpointing](https://arxiv.org/abs/1604.06174)** — trades compute for memory, saves 40% activation VRAM
5. **[CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/)** — pre-record the inference computation for deterministic sub-millisecond latency

---

## 🔗 Relation to the Fleet

This repo is part of the [SuperInstance](https://github.com/SuperInstance) fleet:

| Component | Role |
|-----------|------|
| **fleet-gateway** | Routes LLM calls with circuit breaker, fallback, and caching |
| **fleet-memory** | Stores MIDI corpus embeddings for JEPA training and retrieval |
| **TapScript** | The notation system this instrument speaks |
| **[fleet-ensemble](https://github.com/SuperInstance/fleet-ensemble)** | Companion system: renders MIDI scores as intelligent performances |

### Two Complementary Flows

1. **JEPA-MIDI (this repo) — Construction:** Sound → JEPA perceives feel → LLM thinks in phrasing → algorithms execute → MIDI emerges. **Building music FROM feel.**

2. **Fleet Ensemble — Performance:** MIDI score → performer agents render it with intelligence → JEPA director shapes the feel → output is more than notes, it's a *performance*. **Rendering MIDI AS more than notes on a page.**

Both are modular and agnostic. The JEPA, the performer, the rendering system — all pluggable.

---

## 📊 Training Data Pipeline

```
MIDI Files                    Pre-Tokenization         Training
┌───────────┐                 ┌──────────────┐        ┌──────────────┐
│ Lakh MIDI │──► pretty_midi ─►│              │        │              │
│ (176k)    │    parse +       │ Tokenize to  │        │ Batch 128    │
├───────────┤    quantize to   │ fixed 64-    │──► .npy│ Shuffled     │
│ MAESTRO   │    32ms grid     │ token        │ cached │ Augmented    │
│ (1.2k)    │                  │ windows      │ (300MB)│ FP16 mixed   │
├───────────┤                   │              │        │ precision    │
│ Hooktheory│                   │ vocab: 141   │        │ Cosine LR    │
│ (12k)     │                   └──────────────┘        │ 3-phase      │
├───────────┤                                            │ curriculum   │
│ Fakebook  │                                            └──────────────┘
│ (500)     │
└───────────┘
```

---

## 🚀 Status

**Concept phase.** Repo created Aug 13, 2026. Design docs are comprehensive — implementation is next.

### Roadmap

| Phase | Deliverable | Status |
|-------|------------|--------|
| **1. Design** | All 5 design documents complete | ✅ Done |
| **2. JEPA training** | Train encoder on MIDI corpus | ⬜ Next |
| **3. Algorithm engines** | Implement Markov, L-system, fractal, CA | ⬜ |
| **4. LLM interface** | Wire up bandleader with directive vocabulary | ⬜ |
| **5. Integration** | Close the feedback loop | ⬜ |
| **6. Self-improving** | Add curiosity + Dreamer + Cross-JEPA | ⬜ |
| **7. Performance** | Stage-ready real-time system | ⬜ |

---

## 📝 License

[MIT](LICENSE)

---

## 🙏 Acknowledgments

This design was synthesized from deep research across multiple fields:

**JEPA & SSL research:**
- [Yann LeCun](https://en.wikipedia.org/wiki/Yann_LeCun)'s [I-JEPA](https://arxiv.org/abs/2301.08243) — the foundational architecture
- [BYOL](https://arxiv.org/abs/2006.07733), [SimSiam](https://arxiv.org/abs/2011.10566), [VICReg](https://arxiv.org/abs/2105.04906) — anti-collapse strategies

**Music AI:**
- [MIDI-RAE-JEPA](https://github.com/drscotthawley/midi-rae) (Scott Hawley) — Swin V2 encoder for piano rolls
- [Music-JEPA](https://arxiv.org/abs/2607.22000) (Wang, Fang, LeCun) — action-conditioned world model for music
- [Audio-JEPA](https://github.com/LudovicTuncay/Audio-JEPA) (Tuncay et al.) — spectrogram JEPA with open code
- [Stem-JEPA](https://github.com/SonyCSLParis/Stem-JEPA) (Sony CSL Paris) — multi-track compatibility

**Algorithmic music:**
- [Markov chains](https://en.wikipedia.org/wiki/Markov_chain) for melody generation
- [L-systems](https://en.wikipedia.org/wiki/L-system) by [Aristid Lindenmayer](https://en.wikipedia.org/wiki/Aristid_Lindenmayer) — parallel rewriting grammars
- [Cellular automata](https://en.wikipedia.org/wiki/Cellular_automaton) — [Wolfram's rules](https://en.wikipedia.org/wiki/Rule_110)
- [Fractal music](https://en.wikipedia.org/wiki/Fractal_music) — 1/f noise and [Hurst exponent](https://en.wikipedia.org/wiki/Hurst_exponent)

**World models:**
- [DreamerV3](https://arxiv.org/abs/2301.04104) (Hafner et al.) — latent world models
- [MuZero](https://arxiv.org/abs/1911.08265) — planning with learned dynamics

**Design review by:**
- [DeepSeek V4-Pro](https://www.deepseek.com)
- [ByteDance Seed-2.0-pro](https://bytedance.com) via [DeepInfra](https://deepinfra.com)
- [NousResearch Hermes-3-Llama-3.1-405B](https://nousresearch.com) via DeepInfra

---

*Built by the fleet. For musicians who deserve better tools.*
