# Self-Improving Harnesses for JEPA-MIDI

## How the system learns to be a better musician — by playing

> *A great musician doesn't just practice scales. They chase the unknown. They find the edge of what they can hear, and they lean over it. The question is: can a machine learn to do the same? Can a JEPA-MIDI system develop **curiosity** — an intrinsic drive to explore musical territories where its predictions fail?*
>
> *This document describes four architectures that make the system not just play music, but **learn from playing**. Each architecture is a self-improvement loop — a harness that tightens the feedback between perception and generation until the system starts teaching itself.*

---

## Table of Contents

1. [The Self-Improvement Problem](#1-the-self-improvement-problem)
2. [Architecture 1: Intrinsic Curiosity & Novelty Loops](#2-architecture-1-intrinsic-curiosity--novelty-loops)
3. [Architecture 2: Adversarial Context Masking](#3-architecture-2-adversarial-context-masking)
4. [Architecture 3: Latent Action World Modeling (Dreamer)](#4-architecture-3-latent-action-world-modeling-dreamer)
5. [Architecture 4: Multi-Modal Coherence Alignment (Cross-JEPA)](#5-architecture-4-multi-modal-coherence-alignment-cross-jepa)
6. [Integration with the 3-Layer System](#6-integration-with-the-3-layer-system)
7. [Mathematical Foundations](#7-mathematical-foundations)
8. [Implementation Pseudocode](#8-implementation-pseudocode)
9. [Hardware Constraints & Performance Targets](#9-hardware-constraints--performance-targets)
10. [Paper References & Further Reading](#10-paper-references--further-reading)

---

## 1. The Self-Improvement Problem

### Why Static Training Isn't Enough

The base JEPA-MIDI system described in the [training design document](jepa-training-design.md) learns from a fixed corpus: [Lakh MIDI](https://colinraffel.com/projects/lmd/), [MAESTRO](https://magenta.tensorflow.org/datasets/maestro), Hooktheory. It learns to perceive music — to embed it in a 384-dimensional space where similar-feeling pieces cluster together. This is powerful. But it's also **frozen**.

A musician who only listens to recordings never improves. They need to **play**, hear themselves, and adjust. The same is true for the JEPA-MIDI system. The three layers — [LLM bandleader](llm-interface-design.md), JEPA perceiver, and [algorithmic engines](agentic-algorithmic-music.md) — form a closed loop. But without a learning signal flowing back through that loop, the system repeats the same patterns indefinitely. It plays competently but never grows.

### The Four Harnesses

Each architecture below is a **self-improvement loop** — a way for the system to learn from its own output, without human labels or external rewards:

| Harness | Core Idea | Analogy | Learns |
|---------|-----------|---------|--------|
| **[Intrinsic Curiosity](#2-architecture-1-intrinsic-curiosity--novelty-loops)** | Prediction error = reward | Jazz musician chasing "the unknown" | What musical territories are worth exploring |
| **[Adversarial Masking](#3-architecture-2-adversarial-context-masking)** | Co-evolutionary game | Boxer training against a sparring partner who keeps finding new weaknesses | To predict from increasingly sparse context |
| **[Latent Action World Model](#4-architecture-3-latent-action-world-modeling-dreamer)** | Dream the future before playing | Chess master visualizing 5 moves ahead | The causal consequences of musical decisions |
| **[Cross-JEPA Coherence](#5-architecture-4-multi-modal-coherence-alignment-cross-jepa)** | Align music with other modalities | Dancer matching a musician's energy | How music aligns with visuals, movement, other tracks |

### What "Self-Improving" Means Here

These are **not** fine-tuning loops that update the base JEPA encoder during performance. The base encoder (trained per the [training design](jepa-training-design.md)) stays frozen during real-time operation. Instead:

- **Offline self-improvement:** The harnesses run during dedicated training sessions, using recorded performances as data
- **Online adaptation:** Lightweight adapter layers (LoRA, [prompt tuning](https://arxiv.org/abs/2104.08691)) can be updated during performance, modulating the frozen base
- **Curriculum generation:** The harnesses generate training data that targets the system's weaknesses — a self-made curriculum

Think of it this way: the base JEPA is the musician's ear training. The harnesses are what happens when they sit in a room with other musicians and **play** — discovering edges, failing, and trying again.

---

## 2. Architecture 1: Intrinsic Curiosity & Novelty Loops

### The Big Idea

> *When you can't predict what comes next, lean in. Something interesting is happening.*

[Intrinsic curiosity](https://pathak22.github.io/novelty/) is the idea that **prediction error itself** can serve as a reward signal. Instead of external rewards (a human saying "good solo"), the system generates its own internal motivation: the drive to explore musical territories where its predictions fail.

This is directly inspired by [Pathak et al.'s Intrinsic Curiosity Module (ICM)](https://arxiv.org/abs/1705.05363), which uses prediction error in a learned feature space as intrinsic reward for RL agents. For music, the adaptation is natural: the JEPA *already* predicts future embeddings. When those predictions are wrong, that's musically interesting.

### How It Maps to Music

Consider a jazz musician on stage. They've played a ii-V-I a thousand times. They can predict it perfectly. But then the pianist throws in a [tritone substitution](https://en.wikipedia.org/wiki/Tritone_substitution) they've never heard. The musician's internal prediction fails — and that failure is *electric*. It's the moment they lean forward, ears sharp, trying to understand. That prediction error is curiosity.

For the JEPA-MIDI system:

```
Prediction Error ↑  →  Curiosity Reward ↑  →  Explore This Territory
Prediction Error ↓  →  Boredom (low reward)  →  Move On
```

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                INTRINSIC CURIOSITY LOOP                      │
│                                                             │
│   Algorithmic Engines ──► Musical Output                    │
│         ▲                       │                          │
│         │                       ▼                          │
│         │              ┌───────────────┐                   │
│         │              │  JEPA Encoder  │                   │
│         │              │  (frozen)      │                   │
│         │              └───────┬───────┘                   │
│         │                      │ actual embedding           │
│         │                      ▼                            │
│         │              ┌───────────────┐                   │
│         │              │  JEPA Predictor│── predicted ──┐   │
│         │              │  (frozen)      │   embedding   │   │
│         │              └───────────────┘               │   │
│         │                      │                        │   │
│         │                      ▼                        ▼   │
│         │              ┌─────────────────────────────────┐ │
│         │              │   Prediction Error              │ │
│         │              │   ‖z_actual - z_predicted‖²    │ │
│         │              └───────────────┬─────────────────┘ │
│         │                              │                   │
│         │                              ▼                   │
│         │              ┌─────────────────────────────────┐ │
│         │              │   Curiosity Reward              │ │
│         │              │   r = clip(error, 0, 1)         │ │
│         │              │   · novelty_bonus               │ │
│         │              └───────────────┬─────────────────┘ │
│         │                              │                   │
│         │                              ▼                   │
│         │              ┌─────────────────────────────────┐ │
│         │              │   Exploration Policy            │ │
│         │              │   (modulates engine params      │ │
│         │              │    toward high-curiosity zones) │ │
│         │              └───────────────┬─────────────────┘ │
│         │                              │                   │
│         └──────────────────────────────┘                   │
│                                                            │
└─────────────────────────────────────────────────────────────┘
```

### The Math

The curiosity reward at time step *t* is the squared [L2 distance](https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm) between the predicted and actual embeddings:

$$r_t^{curiosity} = \frac{1}{d} \| \hat{z}_{t+1} - z_{t+1} \|_2^2$$

where:
- $\hat{z}_{t+1} = f_{predict}(z_t, \text{context}_t)$ is the JEPA predictor's output
- $z_{t+1} = f_{encode}(\text{actual bars } t+1 \text{ to } t+k)$ is the actual embedding
- $d$ is the embedding dimension (384)

To prevent the system from seeking pure noise (which maximizes prediction error but isn't musically interesting), we apply an **entropy gate**:

$$r_t = \begin{cases} r_t^{curiosity} & \text{if } H(\text{output}_t) < H_{max} \\ 0 & \text{otherwise} \end{cases}$$

where $H$ is the [Shannon entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) of the output distribution. This ensures the system seeks *structured* unpredictability, not chaos.

### Exploration Policy

The curiosity reward modulates algorithmic engine parameters through a lightweight policy network:

```python
class CuriosityExplorer:
    """Explores musical territories using JEPA prediction error as reward."""
    
    def __init__(self, d_embed=384):
        # Lightweight policy: embedding → parameter deltas
        self.policy = nn.Sequential(
            nn.Linear(d_embed, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 32)  # 32 controllable parameters across engines
        )
        self.reward_history = []
        self.novelty_threshold = 0.15
        self.entropy_max = 4.0  # bits — reject pure noise
    
    def compute_reward(self, predicted_embed, actual_embed, output_entropy):
        """Intrinsic curiosity reward."""
        raw_error = F.mse_loss(predicted_embed, actual_embed).item()
        
        # Entropy gate: don't reward pure noise
        if output_entropy > self.entropy_max:
            return 0.0
        
        # Normalize and clip
        reward = min(raw_error / self.novelty_threshold, 1.0)
        self.reward_history.append(reward)
        
        # Decay novelty for repeated exploration of same zone
        # (diminishing returns — same as human curiosity)
        recent_avg = np.mean(self.reward_history[-20:])
        if recent_avg > 0.7 and reward > 0.5:
            reward *= 0.7  # Been here too much — reduce reward
        
        return reward
    
    def suggest_parameters(self, current_embed, current_params):
        """Suggest parameter changes that might lead to interesting music."""
        param_deltas = self.policy(current_embed)
        
        # Scale deltas by recent curiosity level
        curiosity_level = np.mean(self.reward_history[-5:]) if self.reward_history else 0.3
        scale = 0.3 + 0.7 * curiosity_level  # More adventurous when curious
        
        return current_params + scale * param_deltas
```

### Novelty Memory

To avoid cycling through the same "surprising" patterns, the system maintains a **novelty memory** — a buffer of recently visited regions of embedding space:

```python
class NoveltyMemory:
    """Remembers where in embedding space the system has been."""
    
    def __init__(self, capacity=500, d_embed=384):
        self.buffer = deque(maxlen=capacity)
        self.d_embed = d_embed
    
    def is_novel(self, embedding, threshold=0.12):
        """Check if an embedding is in unexplored territory."""
        if len(self.buffer) < 5:
            return True
        
        # Compute distance to nearest neighbor in buffer
        buffer_tensor = torch.stack(list(self.buffer))
        dists = torch.cdist(embedding.unsqueeze(0), buffer_tensor)
        min_dist = dists.min().item()
        
        return min_dist > threshold
    
    def add(self, embedding):
        """Record a visited embedding."""
        self.buffer.append(embedding.detach().cpu())
```

### Connection to Jazz Pedagogy

This architecture formalizes what [jazz educators](https://en.wikipedia.org/wiki/Jazz_education) have taught for a century:

- [**"Play what you hear"**](https://www.jazzadvice.com/articles/play-what-you-hear/) → The JEPA encodes "hearing" as embedding
- [**"Go outside, then come back"**](https://en.wikipedia.org/wiki/Outside_playing) → High-curiosity zones are "outside"; the entropy gate ensures you come back
- [**"Don't play the same solo twice"**](https://en.wikipedia.org/wiki/Jazz_improvisation) → Novelty memory prevents repetition
- **"Leave space"** → Low-density regions can be novel too (silence as exploration)

### Connection to Reinforcement Learning

The curiosity loop is a form of [intrinsic motivation in RL](https://arxiv.org/abs/1705.05363). The system is the agent, the embedding space is the environment, and curiosity reward drives exploration without external labels. The key difference from standard RL: there is no terminal state and no external reward — the system explores perpetually, driven by its own prediction gaps.

For a rigorous treatment, see:
- [Pathak et al., "Curiosity-driven Exploration by Self-Supervised Prediction" (ICM)](https://arxiv.org/abs/1705.05363)
- [Burda et al., "Large-Scale Study of Curiosity-Driven Learning"](https://arxiv.org/abs/1808.04355)
- [Schmidhuber, "A Possibility for Implementing Curiosity and Boredom"](http://people.idsia.ch/~juergen/curiosity.html) — the original 1991 curiosity paper

---

## 3. Architecture 2: Adversarial Context Masking

### The Big Idea

> *Train against an opponent who keeps finding your blind spots. When they stop finding them, you've mastered the art.*

In the base [JEPA training design](jepa-training-design.md), we use **fixed future-block masking**: always mask the last 32 of 64 tokens. This is musically meaningful but has a ceiling — the encoder eventually learns to predict from this specific mask and stops improving.

[Adversarial masking](https://arxiv.org/abs/2208.04333) replaces the fixed mask with a **learned adversary**: a separate network whose job is to find the masking strategy that maximally stumps the encoder. The encoder must then improve to handle harder masks. The masker must then find harder masks. This is a [co-evolutionary game](https://en.wikipedia.org/wiki/Co-evolutionary_learning) — the same dynamic that makes [GANs](https://arxiv.org/abs/1406.2661) work, applied to masking strategy.

### The Minimax Game

$$\min_\theta \max_\phi \mathcal{L}_{JEPA}(\theta, \phi) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \| f_\theta(\text{mask}_\phi(x)) - f_{EMA}(x_{target}) \|_2^2 \right]$$

where:
- $\theta$ = encoder parameters (minimize loss — want good predictions)
- $\phi$ = masker parameters (maximize loss — want hard masks)
- $\text{mask}_\phi(x)$ = the adversarial mask applied to the context
- $f_{EMA}$ = [exponential moving average](https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average) target encoder (stop-gradient)

The encoder learns to predict from sparse context. The masker learns to remove the tokens that are most informative — forcing the encoder to infer them from structure alone.

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│              ADVERSARIAL MASKING GAME                         │
│                                                              │
│  Input Window (64 tokens)                                   │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────────┐                                          │
│  │  Masker Net  │──► adversarial mask (which tokens        │
│  │  (trainable) │    to hide from the encoder)              │
│  └──────┬───────┘                                          │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Context     │    │  Encoder     │    │  Predictor   │  │
│  │  (masked)    │──►│  (trainable)  │──►│  (linear)    │  │
│  └──────────────┘    └──────┬───────┘    └──────┬───────┘  │
│                             │                    │          │
│                             │   predicted embed  │          │
│                             │                    ▼          │
│                             │           ┌──────────────┐   │
│                             │           │ Target (EMA) │   │
│                             │           │   embed      │   │
│                             │           └──────┬───────┘   │
│                             │                  │           │
│                             ▼                  ▼           │
│                      ┌──────────────────────────────┐     │
│                      │     JEPA Loss                │     │
│                      │  L = ‖pred - target‖²        │     │
│                      └──────────┬───────────────────┘     │
│                                 │                          │
│                    ┌────────────┴────────────┐             │
│                    │                         │             │
│                    ▼                         ▼             │
│              Encoder updates            Masker updates     │
│              (∇_θ minimize)            (∇_φ maximize)      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Masker Network

The masker is a small policy network that decides which tokens to mask:

```python
class AdversarialMasker(nn.Module):
    """Learns to find the hardest masking strategy for the encoder."""
    
    def __init__(self, seq_len=64, d_model=384, min_mask=16, max_mask=48):
        super().__init__()
        self.seq_len = seq_len
        self.min_mask = min_mask
        self.max_mask = max_mask
        
        # Scoring network: token embeddings → mask probabilities
        self.scorer = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 1)  # per-token mask score
        )
        
        # Temperature for Gumbel-Softmax sampling
        self.temperature = 1.0
    
    def forward(self, token_embeddings, target_encoder=None):
        """
        Generate an adversarial mask.
        
        Args:
            token_embeddings: [B, seq_len, d_model] from token embedding layer
        Returns:
            mask: [B, seq_len] boolean — True = masked (hidden from encoder)
        """
        scores = self.scorer(token_embeddings).squeeze(-1)  # [B, seq_len]
        
        # Determine number of tokens to mask (random within bounds)
        n_mask = torch.randint(self.min_mask, self.max_mask + 1, (1,)).item()
        
        # Differentiable top-k masking via Gumbel-Softmax
        # (straight-through estimator for gradient flow to masker)
        if self.training:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
            logits = (scores + gumbel_noise) / self.temperature
            mask_probs = torch.softmax(logits, dim=-1)
            
            # Straight-through: hard mask in forward, soft gradient in backward
            top_k_indices = mask_probs.topk(n_mask, dim=-1).indices
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(1, top_k_indices, True)
            
            # STE: attach gradient from mask_probs to the hard mask
            mask STE = mask.float() + mask_probs - mask_probs.detach()
            return mask, mask_probs
        else:
            top_k_indices = scores.topk(n_mask, dim=-1).indices
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(1, top_k_indices, True)
            return mask, None
```

### Training Schedule

The masker and encoder take turns, like [GAN training](https://arxiv.org/abs/1406.2661):

```python
def train_adversarial_jepa(model, masker, dataloader, 
                            enc_steps=5, masker_steps=1):
    """
    Alternating training: encoder improves, then masker adapts.
    
    5 encoder steps per 1 masker step — the encoder needs more
    time to learn from each new masking strategy.
    """
    enc_optim = torch.optim.AdamW(model.online_encoder.parameters(), lr=3e-4)
    masker_optim = torch.optim.AdamW(masker.parameters(), lr=1e-4)
    
    step = 0
    for batch in dataloader:
        tokens = batch.to(device)
        
        if step % (enc_steps + masker_steps) < enc_steps:
            # === ENCODER STEP: minimize JEPA loss ===
            model.train()
            masker.eval()
            
            with torch.no_grad():
                mask, _ = masker(token_embeddings)
            
            pred_embed, target_embed = model(tokens, mask=mask)
            loss = jepa_loss(pred_embed, target_embed)
            
            enc_optim.zero_grad()
            loss.backward()
            enc_optim.step()
            model.update_target()  # EMA update
            
        else:
            # === MASKER STEP: maximize JEPA loss ===
            model.eval()
            masker.train()
            
            with torch.no_grad():
                # Don't update encoder — just measure loss
                pred_embed, target_embed = model(tokens, mask=mask)
                enc_loss = jepa_loss(pred_embed, target_embed)
            
            masker_optim.zero_grad()
            (-enc_loss).backward()  # Maximize encoder's loss
            masker_optim.step()
        
        step += 1
```

### What the Masker Learns

Empirically (from adversarial masking research in NLP and vision), the masker discovers semantically meaningful patterns:

| Musical Token Type | What Masker Targets | Why |
|-------------------|--------------------|-----|
| **[Bar boundaries](https://en.wikipedia.org/wiki/Bar_(music))** | Hides structural markers | Forces encoder to infer phrase structure from content alone |
| **[Strong beats](https://en.wikipedia.org/wiki/Beat_(music))** | Hides downbeat tokens | Encoder must infer pulse from rhythm patterns |
| **[Chord tones](https://en.wikipedia.org/wiki/Chord_tone)** | Hides root/fifth | Encoder must infer harmony from melody contour |
| **[Rest tokens](https://en.wikipedia.org/wiki/Rest_(music))** | Hides silence | Encoder must infer phrasing from note placement |
| **[Velocity extremes](https://en.wikipedia.org/wiki/Dynamics_(music))** | Hides dynamics | Encoder must infer energy from density |

This is a **music-theoretic curriculum** discovered automatically by the adversarial game.

### Connection to Curriculum Learning

[Curriculum learning](https://en.wikipedia.org/wiki/Curriculum_learning) — training on increasingly difficult examples — is typically implemented by a human-designed schedule. Adversarial masking **automatically discovers** the curriculum by finding the masking patterns that exploit the encoder's current weaknesses. The masker is a curriculum designer.

For more on adversarial masking in SSL:
- [Adversarial Masking for Self-Supervised Learning (FasterMAC)](https://arxiv.org/abs/2208.04333)
- [Adversarial Self-Supervised Learning](https://arxiv.org/abs/2006.07546)
- [Learning to Pretrain by Self-Supervised Contrastive Learning](https://arxiv.org/abs/2106.12653)

---

## 4. Architecture 3: Latent Action World Modeling (Dreamer)

### The Big Idea

> *Before you play the note, hear it in your head. Before you change the rule, dream what it sounds like.*

[DreamerV3](https://danijar.com/project/dreamerv3/) ([Hafner et al., 2023](https://arxiv.org/abs/2301.04104)) is a [world model](https://en.wikipedia.org/wiki/World_model_(artificial_intelligence)) that learns to **dream** — to simulate the future in a compressed latent space before acting. An agent imagines possible futures, evaluates them, and chooses the action that leads to the best imagined outcome.

For music, the translation is exquisite:

| DreamerV3 Concept | JEPA-MIDI Equivalent |
|-------------------|---------------------|
| **State** $s_t$ | JEPA embedding of current music (what it sounds like right now) |
| **Action** $a_t$ | MIDI parameter changes (tempo, density, algorithm rules, key) |
| **Dynamics model** $f(s_t, a_t) \to s_{t+1}$ | What will it sound like after parameter changes? |
| **Reward** $r_t$ | Musical quality (JEPA coherence + [curiosity](#2-architecture-1-intrinsic-curiosity--novelty-loops)) |
| **Dream** | Imagine musical futures before playing them |

### The Recurrent State-Space Model (RSSM)

DreamerV3's core is the [RSSM](https://arxiv.org/abs/1912.01603) — a [recurrent neural network](https://en.wikipedia.org/wiki/Recurrent_neural_network) that maintains a **stochastic latent state** over time:

$$h_t = f_\theta(h_{t-1}, s_{t-1}, a_{t-1})$$

$$s_t \sim q_\theta(s_t | h_t, o_t) \quad \text{(posterior, during training)}$$

$$s_t \sim p_\theta(s_t | h_t) \quad \text{(prior, during dreaming)}$$

where:
- $h_t$ = deterministic hidden state (the "memory")
- $s_t$ = stochastic latent state (the "imagination" — multiple possible futures)
- $a_{t-1}$ = the MIDI parameter action taken
- $o_t$ = the JEPA embedding observation

During **training**, the model has access to the actual musical output ($o_t$), so it uses the posterior $q$ (which sees the observation). During **dreaming**, there is no actual output yet — the model uses the prior $p$ (which imagines from the hidden state alone). This is how the system **hallucinates musical futures**.

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│              LATENT ACTION WORLD MODEL                        │
│                                                              │
│   ┌─────────┐  Action    ┌───────────┐  Next State          │
│   │ Current │──(MIDI ──►│  RSSM     │──► ŝ_{t+1}           │
│   │ State   │  params)   │  Dynamics │    (dreamed future)  │
│   │ s_t     │◄──────────│  Model    │                      │
│   └────┬────┘           └─────┬─────┘                      │
│        │                       │                             │
│        │ JEPA embed            │ Predicted embed              │
│        │ (observed)            │ (imagined)                   │
│        ▼                       ▼                             │
│   ┌──────────┐           ┌──────────┐                      │
│   │ Actual   │           │ Imagined │                       │
│   │ Music    │◄──────────│ Future   │                       │
│   │ Output   │  compare  │ Quality  │                       │
│   └──────────┘           └──────────┘                      │
│                                                              │
│   DURING TRAINING:                                          │
│     Loss = reconstruction + dynamics + reward prediction    │
│     The model learns f(state, action) → next_state           │
│                                                              │
│   DURING PERFORMANCE:                                       │
│     The system "dreams" k steps ahead:                      │
│     1. Imagine parameter change                             │
│     2. Predict resulting embedding                          │
│     3. Evaluate musical quality                             │
│     4. Choose best action                                   │
│     5. Execute on real engines                              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### MIDI Parameters as Actions

The action space for the world model is the set of all controllable parameters across the algorithmic engines (defined in the [agentic algorithmic music document](agentic-algorithmic-music.md)):

```python
class MIDIActionSpace:
    """The action space for the Dreamer world model."""
    
    def __init__(self):
        # Continuous parameters (normalized to [0, 1])
        self.markov_params = {
            'temperature': (0.1, 2.0),       # sampling randomness
            'order': (1, 5),                 # memory depth
            'chromatic_tolerance': (0.0, 1.0),
            'syncopation': (0.0, 1.0),
            'swing': (0.0, 0.75),
        }
        self.lsystem_params = {
            'rule_complexity': (1, 10),
            'stochasticity': (0.0, 1.0),
            'expansion_depth': (3, 8),
        }
        self.fractal_params = {
            'hausdorff_D': (1.0, 2.0),       # fractal dimension
            'lacunarity': (0.0, 1.5),         # gap distribution
            'hurst_exponent': (0.0, 1.0),     # long-range dependence
        }
        self.ca_params = {
            'rule': (0, 255),                # Wolfram rule number (discrete)
            'density_target': (0.1, 0.8),
        }
        self.global_params = {
            'tempo': (60, 200),
            'energy': (0.0, 1.0),
            'tension': (0.0, 1.0),
        }
    
    def to_vector(self, params_dict):
        """Flatten parameters to a continuous action vector for the world model."""
        vec = []
        for category in [self.markov_params, self.lsystem_params, 
                         self.fractal_params, self.ca_params, self.global_params]:
            for name, (lo, hi) in category.items():
                val = params_dict.get(name, (lo + hi) / 2)
                vec.append((val - lo) / (hi - lo))  # normalize to [0, 1]
        return torch.tensor(vec, dtype=torch.float32)
    
    @property
    def dim(self):
        return 18  # total continuous action dimensions
```

### The Dreaming Loop

```python
class MusicalDreamer:
    """Dreams musical futures and selects the best one to play."""
    
    def __init__(self, rssm, reward_model, action_space, 
                 n_imagination_steps=5, n_candidates=8):
        self.rssm = rssm               # learned dynamics model
        self.reward_model = reward_model  # predicts musical quality
        self.action_space = action_space
        self.n_steps = n_imagination_steps  # how far to dream ahead
        self.n_candidates = n_candidates    # parallel dreams to evaluate
        self.d_embed = 384
    
    def dream_and_choose(self, current_embed, current_params, 
                          target_trajectory=None):
        """
        Dream multiple futures, evaluate them, choose the best action sequence.
        
        This runs BEFORE the LLM makes its directive — it provides the LLM
        with "here's what I think will happen if we do X" information.
        """
        # 1. Sample candidate action sequences (parameter trajectories)
        candidates = []
        for _ in range(self.n_candidates):
            action_seq = self._sample_action_sequence(current_params)
            candidates.append(action_seq)
        
        # 2. Dream forward for each candidate
        dreamed_rewards = []
        for action_seq in candidates:
            reward = self._imagine_trajectory(
                current_embed, action_seq, target_trajectory
            )
            dreamed_rewards.append(reward)
        
        # 3. Select best candidate
        best_idx = np.argmax(dreamed_rewards)
        best_actions = candidates[best_idx]
        
        return best_actions[0]  # Return first action (immediate)
    
    @torch.no_grad()
    def _imagine_trajectory(self, start_embed, action_seq, target=None):
        """Dream forward through the RSSM, accumulate predicted reward."""
        h = self.rssm.init_hidden(start_embed)
        s = start_embed
        total_reward = 0.0
        discount = 1.0
        
        for action in action_seq:
            # RSSM forward step: imagine next state
            h = self.rssm.deterministic_step(h, s, action)
            s_prior = self.rssm.prior(h)  # imagined embedding (no observation)
            s = s_prior.sample()
            
            # Predict reward (musical quality)
            r = self.reward_model(s, action)
            total_reward += discount * r
            discount *= 0.95  # gamma = 0.95
        
        # Bonus: alignment with target trajectory
        if target is not None:
            alignment = -F.mse_loss(s, target)
            total_reward += 0.3 * alignment
        
        return total_reward
```

### The RSSM Architecture

```python
class MusicalRSSM(nn.Module):
    """Recurrent State-Space Model for musical dynamics."""
    
    def __init__(self, d_embed=384, d_action=18, d_hidden=512):
        super().__init__()
        self.d_embed = d_embed
        self.d_hidden = d_hidden
        
        # Deterministic path: GRU
        self.gru = nn.GRUCell(d_embed + d_action, d_hidden)
        
        # Prior: p(s_t | h_t) — imagination
        self.prior_net = nn.Sequential(
            nn.Linear(d_hidden, 256),
            nn.GELU(),
            nn.Linear(256, 2 * d_embed)  # mean, log_var
        )
        
        # Posterior: q(s_t | h_t, o_t) — observation
        self.posterior_net = nn.Sequential(
            nn.Linear(d_hidden + d_embed, 256),
            nn.GELU(),
            nn.Linear(256, 2 * d_embed)  # mean, log_var
        )
    
    def init_hidden(self, embed):
        """Initialize deterministic state from embedding."""
        return torch.zeros(1, self.d_hidden)
    
    def deterministic_step(self, h, s, a):
        """GRU forward step."""
        return self.gru(torch.cat([s, a], dim=-1), h)
    
    def prior(self, h):
        """Sample from prior p(s_t | h_t) — for dreaming."""
        params = self.prior_net(h)
        mean, log_var = params.chunk(2, dim=-1)
        std = torch.exp(0.5 * log_var)
        s = mean + std * torch.randn_like(mean)  # reparameterize
        return Normal(mean, std), s
    
    def posterior(self, h, o):
        """Sample from posterior q(s_t | h_t, o_t) — for training."""
        params = self.posterior_net(torch.cat([h, o], dim=-1))
        mean, log_var = params.chunk(2, dim=-1)
        std = torch.exp(0.5 * log_var)
        s = mean + std * torch.randn_like(mean)
        return Normal(mean, std), s
```

### Training the World Model

The world model trains on recorded performance data:

```python
def train_dreamer(dreamer, performance_recordings, je, n_epochs=50):
    """
    Train the world model on recorded JEPA-MIDI performances.
    
    performance_recordings: list of (embeddings, actions, rewards) tuples
    """
    for epoch in range(n_epochs):
        for embeddings, actions, rewards in performance_recordings:
            T = len(embeddings)
            h = dreamer.rssm.init_hidden(embeddings[0])
            
            total_loss = 0.0
            for t in range(T - 1):
                # Deterministic step
                h = dreamer.rssm.deterministic_step(
                    h, embeddings[t], actions[t]
                )
                
                # Posterior (with observation)
                post_dist, s_post = dreamer.rssm.posterior(h, embeddings[t+1])
                
                # Prior (without observation — what the model expected)
                prior_dist, _ = dreamer.rssm.prior(h)
                
                # KL divergence between prior and posterior
                # (with clipping for stability — DreamerV3 uses 80% clipping)
                kl = kl_divergence(post_dist, prior_dist).mean()
                kl = max(kl.item(), 1.0)  # clip at 1 nat
                
                # Reward prediction loss
                predicted_reward = dreamer.reward_model(s_post, actions[t])
                reward_loss = F.mse_loss(predicted_reward, rewards[t])
                
                # Dynamics loss: can the model predict next embedding?
                pred_embed = dreamer.rssm.prior_net(h)[:, :384]  # mean
                dyn_loss = F.mse_loss(pred_embed, embeddings[t+1])
                
                total_loss += dyn_loss + 0.5 * reward_loss + 0.1 * kl
            
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

### Why This Matters Musically

The Dreamer architecture gives the system something no previous music AI has: **the ability to rehearse the future**. Before changing a rule, the system imagines what the music will sound like. Before deciding to build tension, it dreams the next 8 bars under several parameter choices and picks the one that sounds best.

This is what [Wynton Kelly](https://en.wikipedia.org/wiki/Wynton_Kelly) did when he laid back on the turnaround in "Freddie Freeloader." He heard it in his head first. He dreamed the future, liked it, and played it.

For deeper reading:
- [DreamerV3: Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)
- [DreamerV2: Mastering Atari with Discrete World Models](https://arxiv.org/abs/2010.02193)
- [PlaNet: Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.01651)
- [World Models (Ha & Schmidhuber, 2018)](https://worldmodels.github.io/) — the original
- [MuZero: Mastering Go, Chess, Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265)

---

## 5. Architecture 4: Multi-Modal Coherence Alignment (Cross-JEPA)

### The Big Idea

> *Music doesn't exist in isolation. It syncs with visuals, movement, lights, other instruments. The question isn't "does this sound good?" but "does this feel coherent across every sense?"*

[Cross-JEPA](https://arxiv.org/abs/2310.12747) extends the JEPA framework to multiple modalities: it learns a shared latent space where music, visuals, and movement can be compared for **coherence**. A "critic" network evaluates whether what you hear aligns with what you see.

For the JEPA-MIDI system, this means:
- A music JEPA encodes the audio/MIDI output
- A visual JEPA encodes stage lighting, projections, or dancer movement
- A **coherence critic** evaluates whether they match

When the lights get brighter, does the music get more energetic? When the dancer pauses, does the music breathe? [Cross-modal alignment](https://en.wikipedia.org/wiki/Multimodal_learning) is what makes a performance feel unified rather than disjointed.

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│              CROSS-MODAL COHERENCE ALIGNMENT                  │
│                                                              │
│   ┌─────────────┐         ┌─────────────┐                   │
│   │  Music JEPA │         │ Visual JEPA │                   │
│   │  (frozen)   │         │ (frozen)    │                   │
│   └──────┬──────┘         └──────┬──────┘                   │
│          │                       │                           │
│     z_music                 z_visual                        │
│     [384-dim]              [384-dim]                        │
│          │                       │                           │
│          └───────────┬───────────┘                           │
│                      │                                       │
│                      ▼                                       │
│              ┌───────────────┐                              │
│              │ Cross-JEPA    │                              │
│              │ Coherence     │                              │
│              │ Critic        │                              │
│              │ (trainable)   │                              │
│              └───────┬───────┘                              │
│                      │                                       │
│                      ▼                                       │
│              coherence score [0, 1]                         │
│              + alignment gradient                           │
│                                                              │
│   The critic learns:                                         │
│   - Procrustes alignment between modalities                  │
│   - Temporal lag estimation (music leads? visual leads?)     │
│   - Energy envelope correlation                              │
│   - Emotional valence matching                               │
│                                                              │
│   During performance:                                        │
│   - Low coherence → LLM adjusts music to match visuals       │
│   - High coherence → reinforcement signal                    │
│   - Gradient flows back to music engines via LLM directives  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### The Coherence Critic

The critic learns a [Procrustes alignment](https://en.wikipedia.org/wiki/Procrustes_analysis) — finding the optimal rotation/translation that maps one modality's embedding space onto another's:\n\n$$\min_R \| R \cdot z_{music} - z_{visual} \|_2^2 \quad \text{s.t.} \quad R^T R = I$$

where $R$ is an orthogonal matrix (rotation in 384-dimensional space). This is the [Orthogonal Procrustes problem](https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem), solvable in closed form via [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition).

```python
class CrossJEPACritic(nn.Module):
    """Evaluates coherence between music and visual modalities."""
    
    def __init__(self, d_embed=384):
        super().__init__()
        
        # Learnable projection to align modalities (Procrustes-like)
        self.alignment = nn.Linear(d_embed, d_embed, bias=False)
        
        # Initialize as identity (start aligned)
        nn.init.eye_(self.alignment.weight)
        
        # Coherence estimator
        self.coherence_net = nn.Sequential(
            nn.Linear(d_embed * 3, 512),  # [z_music, z_visual, z_diff]
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Temporal lag estimator (does music lead or follow?)
        self.lag_net = nn.Sequential(
            nn.Linear(d_embed * 2, 256),
            nn.GELU(),
            nn.Linear(256, 1)  # estimated lag in ms (positive = music leads)
        )
    
    def forward(self, z_music, z_visual):
        """
        Compute coherence score and alignment gradient.
        
        Args:
            z_music: [B, d_embed] — music JEPA embedding
            z_visual: [B, d_embed] — visual JEPA embedding
        
        Returns:
            coherence: [B, 1] — 0 = incoherent, 1 = perfectly aligned
            lag_ms: [B, 1] — estimated temporal offset
            gradient: gradient w.r.t. z_music (for feedback to music engines)
        """
        # Align music to visual space
        z_music_aligned = self.alignment(z_music)
        
        # Difference vector (captures misalignment)
        z_diff = z_music_aligned - z_visual
        
        # Coherence score
        coherence = self.coherence_net(
            torch.cat([z_music_aligned, z_visual, z_diff], dim=-1)
        )
        
        # Temporal lag
        lag_ms = self.lag_net(torch.cat([z_music_aligned, z_visual], dim=-1))
        
        return coherence, lag_ms, z_diff
```

### Training the Critic

The critic trains on **aligned and misaligned pairs**:

```python
def train_coherence_critic(critic, music_embeds, visual_embeds, 
                            aligned_pairs, n_epochs=100):
    """
    Train the coherence critic on aligned/misaligned pairs.
    
    Positive examples: (music, visual) from the same timestamp
    Negative examples: (music, visual) from different timestamps
    """
    optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-4)
    
    for epoch in range(n_epochs):
        for music, visual in aligned_pairs:
            # Positive pair
            pos_coherence, _, _ = critic(music.unsqueeze(0), visual.unsqueeze(0))
            pos_loss = -torch.log(pos_coherence + 1e-8)
            
            # Negative pair (shuffle visual)
            neg_visual = visual[torch.randperm(visual.size(0))]
            neg_coherence, _, _ = critic(music.unsqueeze(0), neg_visual.unsqueeze(0))
            neg_loss = -torch.log(1 - neg_coherence + 1e-8)
            
            # Contrastive loss
            loss = pos_loss + neg_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Multi-Modal Inputs

Beyond visuals, the Cross-JEPA can align music with:

| Modality | Input | JEPA Encoder | Use Case |
|----------|-------|-------------|----------|
| **Stage lighting** | DMX values per frame | MLP encoder | Light changes match music dynamics |
| **Dancer motion** | [Pose keypoints](https://en.wikipedia.org/wiki/Pose_estimation) per frame | [ST-GCN](https://arxiv.org/abs/1801.07455) encoder | Music follows choreography |
| **Crowd response** | Audio level / motion from audience | Audio JEPA | System reads the room |
| **Another instrument** | MIDI stream | Same music JEPA | Ensemble coherence |
| **Video/Projections** | Video frames | [V-JEPA](https://arxiv.org/abs/2403.02537) | Score follows picture |

### Connection to Multi-Track Coherence

The most immediate application is **multi-track musical coherence**: ensuring that a bass line, drum pattern, and melody generated by different algorithmic engines feel like they belong together.

This is exactly what [Sony CSL Paris's Stem-JEPA](https://github.com/SonyCSLParis/Stem-JEPA) does — it uses JEPA to estimate musical stem compatibility. The Cross-JEPA critic generalizes this to arbitrary modality pairs.

For more on cross-modal learning:
- [ImageBind: One Embedding Space to Bind Them All](https://arxiv.org/abs/2305.05665)
- [CLIP: Connecting Text and Images](https://arxiv.org/abs/2103.00020)
- [AudioCLIP: Extending CLIP to Audio](https://arxiv.org/abs/2106.13043)
- [Wav2CLIP: Learning Robust Audio Representations](https://arxiv.org/abs/2207.09366)

---

## 6. Integration with the 3-Layer System

### How the Harnesses Connect

Each harness integrates at a specific layer of the JEPA-MIDI system:

```
┌───────────────────────────────────────────────────────────────┐
│                     LLM BANDLEADER                             │
│                                                               │
│   ┌─────────────────────────────────────────────────────┐    │
│   │  Dreamer feeds imagined futures to LLM context       │    │
│   │  "If I increase tempo 5 BPM, the music will feel    │    │
│   │   more urgent in 4 bars. If I drop density instead, │    │
│   │   it will feel more spacious."                       │    │
│   └─────────────────────────────────────────────────────┘    │
│                                                               │
│   ┌─────────────────────────────────────────────────────┐    │
│   │  Cross-JEPA coherence score in sensory context       │    │
│   │  "Visual coherence is 0.45 (low). The lights say     │    │
│   │   'cool down' but the music says 'build.' Adjust."   │    │
│   └─────────────────────────────────────────────────────┘    │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                     JEPA PERCEIVER                             │
│                                                               │
│   ┌─────────────────────────────────────────────────────┐    │
│   │  Adversarial masking runs OFFLINE during training    │    │
│   │  The encoder that survives adversarial masking       │    │
│   │  is stronger at inference time.                      │    │
│   └─────────────────────────────────────────────────────┘    │
│                                                               │
│   ┌─────────────────────────────────────────────────────┐    │
│   │  Curiosity loop runs ONLINE during performance       │    │
│   │  Prediction error modulates exploration policy       │    │
│   │  High error → "something interesting is happening"   │    │
│   │  → suggest adventurous parameters to LLM             │    │
│   └─────────────────────────────────────────────────────┘    │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                  ALGORITHMIC ENGINES                           │
│                                                               │
│   All harness outputs eventually become parameter adjustments │
│   to the Markov/L-system/Fractal/CA engines.                  │
│                                                               │
│   Dreamer → "try these params, I dreamed they'll sound good"  │
│   Curiosity → "try these params, they'll lead somewhere new"  │
│   Cross-JEPA → "try these params, they'll align with visuals" │
│   Adversarial → (offline) stronger base encoder               │
└───────────────────────────────────────────────────────────────┘
```

### Online vs. Offline

| Harness | Mode | Frequency | VRAM Impact |
|---------|------|-----------|-------------|
| **Curiosity Loop** | Online | Every 2 bars (~4s) | +50 MB (policy net) |
| **Dreamer** | Online (lightweight) | Every phrase (~8s) | +200 MB (RSSM + reward) |
| **Cross-JEPA** | Online (if visuals available) | Every 2 bars | +100 MB (critic) |
| **Adversarial Masking** | Offline only | During training sessions | +150 MB (masker net) |

Total online overhead: **~350 MB** on top of the 138 MB inference footprint = **~488 MB**. Well within the 2.8 GB budget.

### The Self-Made Curriculum

The four harnesses combine into a **self-generating training curriculum**:

1. **Adversarial masking** discovers what the encoder can't predict → generates hard training examples
2. **Curiosity loop** discovers what the system hasn't explored → generates novel musical territories
3. **Dreamer** discovers what parameter changes lead to good futures → generates control strategies
4. **Cross-JEPA** discovers what modalities are misaligned → generates coherence targets

Together, they form a closed loop of **perpetual self-improvement**:

```
Adversarial Masking → Stronger Encoder → Better Predictions
Better Predictions → More Accurate Curiosity Signal → Better Exploration
Better Exploration → Richer Performance Data → Better Dreamer Training
Better Dreamer → Better Parameter Choices → More Coherent Output
More Coherent Output → Cross-JEPA Refinement → Stronger Multi-Modal Alignment
```

---

## 7. Mathematical Foundations

### 7.1 Information-Theoretic Curiosity

The curiosity reward is grounded in [information theory](https://en.wikipedia.org/wiki/Information_theory). The prediction error $\| \hat{z} - z \|^2$ approximates the [KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) between the predicted and actual distributions:

$$D_{KL}(q \| p) \approx \frac{1}{2} \| \hat{z} - z \|_2^2$$

when both distributions are approximately [Gaussian](https://en.wikipedia.org/wiki/Multivariate_normal_distribution) with unit variance. Thus, curiosity reward ≈ [information gain](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees) — the system is rewarded for visiting states where it learns the most new information.

Reference: [Still & Precup, "An Information-Theoretic Approach to Curiosity-Driven Reinforcement Learning"](https://arxiv.org/abs/1106.08089).

### 7.2 Adversarial Minimax as Two-Player Game

The adversarial masking game is a [two-player zero-sum game](https://en.wikipedia.org/wiki/Zero-sum_game) in the [von Neumann](https://en.wikipedia.org/wiki/Minimax_theorem) sense:

$$\min_\theta \max_\phi \mathcal{L}(\theta, \phi)$$

Under appropriate conditions (concavity in $\phi$, convexity in $\theta$), this has a [saddle-point equilibrium](https://en.wikipedia.org/wiki/Saddle_point) — a strategy pair $(\theta^*, \phi^*)$ where neither player benefits from deviating. In practice, we use [alternating gradient descent/ascent](https://arxiv.org/abs/1406.2661), the same algorithm that makes [GANs](https://en.wikipedia.org/wiki/Generative_adversarial_network) work.

### 7.3 Dreamer's ELBO Objective

DreamerV3 optimizes a [variational lower bound (ELBO)](https://en.wikipedia.org/wiki/Evidence_lower_bound):

$$\mathcal{L}_{Dreamer} = \underbrace{\mathbb{E}_q[\log p(o_t | s_t)]}_{\text{reconstruction}} - \underbrace{D_{KL}(q(s_t | h_t, o_t) \| p(s_t | h_t))}_{\text{KL regularization}} + \underbrace{\mathbb{E}[\log p(r_t | s_t)]}_{\text{reward prediction}}$$

The KL term keeps the prior (imagination) close to the posterior (observation) — teaching the model to dream accurately. Without it, the model's dreams would diverge from reality.

Reference: [Hafner et al., "Understanding Latent World Models" (DreamerV3 appendix)](https://arxiv.org/abs/2301.04104).

### 7.4 Procrustes Alignment for Cross-JEPA

The [Orthogonal Procrustes problem](https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem) finds the best rotation $R$ to align two sets of vectors:

$$R^* = \arg\min_R \| RA - B \|_F^2 \quad \text{s.t.} \quad R^T R = I$$

Solution: if $A^T B = U \Sigma V^T$ (via [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition)), then $R^* = UV^T$.

This gives a closed-form initialization for the Cross-JEPA alignment matrix, which is then fine-tuned end-to-end.

---

## 8. Implementation Pseudocode

### Complete Self-Improvement Loop

```python
class SelfImprovingHarness:
    """
    All four harnesses integrated into the JEPA-MIDI system.
    
    Runs during performance (online) and during dedicated training (offline).
    """
    
    def __init__(self, jepa_encoder, llm_interface, engines):
        # Core system
        self.encoder = jepa_encoder  # Frozen JEPA encoder
        self.llm = llm_interface     # LLM bandleader
        self.engines = engines       # Algorithmic engines
        
        # Harness 1: Curiosity
        self.curiosity = CuriosityExplorer(d_embed=384)
        self.novelty_memory = NoveltyMemory(capacity=500)
        
        # Harness 2: Adversarial (offline only)
        self.masker = AdversarialMasker(seq_len=64, d_model=384)
        self.adversarial_active = False  # Only during training
        
        # Harness 3: Dreamer
        self.rssm = MusicalRSSM(d_embed=384, d_action=18)
        self.reward_model = MusicalRewardPredictor(d_embed=384, d_action=18)
        self.dreamer = MusicalDreamer(self.rssm, self.reward_model, 
                                       MIDIActionSpace())
        
        # Harness 4: Cross-JEPA (optional, if visual modality available)
        self.visual_encoder = None  # Set if visuals available
        self.coherence_critic = CrossJEPACritic(d_embed=384)
    
    def perceive(self, midi_output):
        """Encode current musical output."""
        tokens = self._midi_to_tokens(midi_output)
        with torch.no_grad():
            embedding = self.encoder(tokens).mean(dim=1)
        return embedding
    
    def predict_next(self, current_embedding):
        """Predict what the next phrase will sound like."""
        with torch.no_grad():
            # JEPA base predictor
            predicted = self.encoder.predictor(current_embedding)
            
            # Dreamer imagination (if trained)
            dreamed = self.rssm.prior(
                self.rssm.deterministic_step(
                    self.rssm.hidden, current_embedding, 
                    self.last_action
                )
            ).sample()
            
            # Blend: JEPA prediction (fast) + Dreamer (if confident)
            dreamer_confidence = self.rssm.confidence(dreamed)
            alpha = dreamer_confidence * 0.4
            blended = (1 - alpha) * predicted + alpha * dreamed
            
        return blended
    
    def compute_curiosity(self, predicted, actual):
        """Harness 1: Curiosity reward."""
        is_novel = self.novelty_memory.is_novel(actual)
        reward = self.curiosity.compute_reward(
            predicted, actual, output_entropy=self._compute_entropy(actual)
        )
        if is_novel:
            self.novelty_memory.add(actual)
        return reward, is_novel
    
    def dream_futures(self, current_embed, current_params):
        """Harness 3: Dream multiple musical futures."""
        return self.dreamer.dream_and_choose(
            current_embed, current_params
        )
    
    def check_coherence(self, music_embed, visual_embed=None):
        """Harness 4: Cross-modal coherence."""
        if visual_embed is None or self.visual_encoder is None:
            return None, None
        coherence, lag, gradient = self.coherence_critic(
            music_embed, visual_embed
        )
        return coherence.item(), gradient
    
    def performance_step(self, midi_output, visual_frame=None):
        """One step of the self-improving loop during performance."""
        # 1. Perceive
        actual_embed = self.perceive(midi_output)
        
        # 2. Check curiosity
        predicted_embed = self.predict_next(actual_embed)
        curiosity, is_novel = self.compute_curiosity(
            predicted_embed, actual_embed
        )
        
        # 3. Dream futures
        dreamed_params = self.dream_futures(
            actual_embed, self.engines.get_params()
        )
        
        # 4. Check cross-modal coherence
        visual_embed = None
        if visual_frame is not None and self.visual_encoder:
            visual_embed = self.visual_encoder(visual_frame)
        coherence, _ = self.check_coherence(actual_embed, visual_embed)
        
        # 5. Build context for LLM
        context = {
            'embedding': actual_embed,
            'curiosity_reward': curiosity,
            'is_novel_territory': is_novel,
            'dreamed_suggestion': dreamed_params,
            'cross_modal_coherence': coherence,
            'predicted_next': predicted_embed,
        }
        
        return context
    
    def offline_training_step(self, performance_recording):
        """
        Harness 2 + 3 offline training on recorded performance.
        
        Called between performances to improve the system.
        """
        # Adversarial masking: strengthen the encoder
        if self.adversarial_active:
            self._train_adversarial_step(performance_recording)
        
        # Dreamer: improve world model on real data
        self._train_dreamer_step(performance_recording)
        
        # Cross-JEPA: improve coherence on aligned pairs
        if self.visual_encoder:
            self._train_coherence_step(performance_recording)
```

---

## 9. Hardware Constraints & Performance Targets

### RTX 4050 Laptop (6 GB VRAM)

| Component | VRAM | Notes |
|-----------|------|-------|
| Base JEPA encoder (frozen) | 36.6 MB | From [training design](jepa-training-design.md) |
| Curiosity policy net | 0.5 MB | 3-layer MLP |
| Novelty memory (500 × 384) | 0.8 MB | Ring buffer |
| Adversarial masker | 2.1 MB | Small scoring net |
| RSSM (Dreamer) | 8.4 MB | GRU + prior + posterior |
| Reward model | 1.2 MB | 2-layer MLP |
| Cross-JEPA critic | 3.6 MB | Linear + MLP |
| **Total harness overhead** | **~17 MB** | **0.6% of VRAM budget** |
| **Total with base system** | **~155 MB** | **5.5% of VRAM budget** |

The harnesses are **negligibly lightweight** compared to the base system. The compute cost is also small: the curiosity loop adds <1ms per step, the Dreamer adds <5ms per dreaming episode, and the Cross-JEPA critic adds <1ms per coherence check.

### Performance Targets

| Operation | Latency Budget | Actual (est.) | Margin |
|-----------|---------------|---------------|--------|
| Curiosity computation | 2 ms | 0.8 ms | 60% |
| Dreamer imagination (5 steps × 8 candidates) | 10 ms | 6.5 ms | 35% |
| Cross-JEPA coherence | 2 ms | 0.5 ms | 75% |
| **Total online overhead** | **15 ms** | **8 ms** | **47%** |

All harnesses operate well within the 125 ms pulse budget.

### Training Targets (Offline)

| Harness | Training Time | Data Required | Improvement Metric |
|---------|--------------|---------------|-------------------|
| Adversarial Masking | 4 hours (on top of base 11h) | Same MIDI corpus | R@10 ↑ 3-5% |
| Dreamer | 6 hours | 100 recorded performances | Prediction accuracy > 0.75 |
| Curiosity (policy net) | 2 hours | Self-generated | Coverage of embedding space > 70% |
| Cross-JEPA critic | 3 hours | 500 aligned (music, visual) pairs | Coherence AUC > 0.85 |

---

## 10. Paper References & Further Reading

### Core Architectures

| Paper | Year | Relevance | Link |
|-------|------|-----------|------|
| **I-JEPA** (Assran et al.) | 2023 | Foundational JEPA architecture | [arXiv:2301.08243](https://arxiv.org/abs/2301.08243) |
| **V-JEPA 2** (Bardes et al.) | 2025 | Action-conditioned JEPA for video | [ai.meta.com/blog/vjepa-2](https://ai.meta.com/blog/vjepa-2/) |
| **DreamerV3** (Hafner et al.) | 2023 | World model architecture for music dreaming | [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) |
| **MuZero** (Schrittwieser et al.) | 2020 | Planning with learned dynamics | [arXiv:1911.08265](https://arxiv.org/abs/1911.08265) |

### Curiosity & Intrinsic Motivation

| Paper | Year | Relevance | Link |
|-------|------|-----------|------|
| **ICM** (Pathak et al.) | 2017 | Curiosity-driven exploration via prediction error | [arXiv:1705.05363](https://arxiv.org/abs/1705.05363) |
| **RND** (Burda et al.) | 2018 | Random network distillation for novelty | [arXiv:1808.04355](https://arxiv.org/abs/1808.04355) |
| **Schmidhuber** | 1991 | Original curiosity-driven learning | [idsia.ch](http://people.idsia.ch/~juergen/curiosity.html) |
| **NGU** (Badia et al.) | 2020 | Never Give Up: curiosity + exploitation | [arXiv:2002.06038](https://arxiv.org/abs/2002.06038) |

### Adversarial Training for SSL

| Paper | Year | Relevance | Link |
|-------|------|-----------|------|
| **GAN** (Goodfellow et al.) | 2014 | Adversarial game foundation | [arXiv:1406.2661](https://arxiv.org/abs/1406.2661) |
| **Adversarial Masking** | 2022 | Adversarial masking for SSL | [arXiv:2208.04333](https://arxiv.org/abs/2208.04333) |
| **FasterMAC** | 2022 | Efficient adversarial masking | [arXiv:2210.06583](https://arxiv.org/abs/2210.06583) |

### Anti-Collapse & Self-Supervised Learning

| Paper | Year | Relevance | Link |
|-------|------|-----------|------|
| **BYOL** (Grill et al.) | 2020 | EMA + predictor (bootstrap your own latent) | [arXiv:2006.07733](https://arxiv.org/abs/2006.07733) |
| **SimSiam** (Chen & He) | 2020 | Stop-gradient prevents collapse | [arXiv:2011.10566](https://arxiv.org/abs/2011.10566) |
| **VICReg** (Bardes et al.) | 2022 | Variance + invariance + covariance regularization | [arXiv:2105.04906](https://arxiv.org/abs/2105.04906) |

### Cross-Modal Alignment

| Paper | Year | Relevance | Link |
|-------|------|-----------|------|
| **CLIP** (Radford et al.) | 2021 | Contrastive cross-modal alignment | [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) |
| **ImageBind** (Girdhar et al.) | 2023 | Bind 6 modalities in one space | [arXiv:2305.05665](https://arxiv.org/abs/2305.05665) |
| **Stem-JEPA** (Riou et al.) | 2024 | Multi-track music stem compatibility | [github.com/SonyCSLParis](https://github.com/SonyCSLParis/Stem-JEPA) |

### Mathematical Foundations

| Topic | Reference | Link |
|-------|-----------|------|
| **KL Divergence** | Wikipedia | [en.wikipedia.org](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) |
| **Orthogonal Procrustes** | Wikipedia | [en.wikipedia.org](https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem) |
| **ELBO** | Wikipedia | [en.wikipedia.org](https://en.wikipedia.org/wiki/Evidence_lower_bound) |
| **Minimax Theorem** | von Neumann, 1928 | [en.wikipedia.org](https://en.wikipedia.org/wiki/Minimax_theorem) |
| **Shannon Entropy** | Wikipedia | [en.wikipedia.org](https://en.wikipedia.org/wiki/Entropy_(information_theory)) |
| **Reparameterization Trick** | Kingma & Welling, 2013 | [arXiv:1312.6114](https://arxiv.org/abs/1312.6114) |
| **Singular Value Decomposition** | Wikipedia | [en.wikipedia.org](https://en.wikipedia.org/wiki/Singular_value_decomposition) |

### Educator Resources

For teaching these concepts:

| Concept | How to Teach It | Resource |
|---------|----------------|----------|
| JEPA | "Show, don't reconstruct" — compare to guessing what's behind a screen vs. drawing it | [LeCun's TED talk](https://www.ted.com/speakers/yann_lecun) |
| Curiosity | Have students improvise music and note when they "get surprised" | [Jazz improvisation exercises](https://www.jazzadvice.com) |
| Adversarial training | Two-student game: one hides clues, the other guesses | [GAN analogy](https://wiki.pathmind.com/generative-adversarial-network-gan) |
| World models | "Dream a song before you play it" — musicians do this naturally | [Music visualization exercises](https://www.youtube.com/results?search_query=music+visualization+exercises) |

---

## Appendix: The Full Self-Improvement Equation

Combining all four harnesses, the total training objective for the self-improving JEPA-MIDI system is:

$$\mathcal{L}_{total} = \underbrace{\mathcal{L}_{JEPA}(\theta, \phi)}_{\text{adversarial base}} + \lambda_1 \underbrace{\mathcal{L}_{Dreamer}(\psi)}_{\text{world model}} + \lambda_2 \underbrace{\mathcal{L}_{coherence}(\omega)}_{\text{cross-modal}} + \lambda_3 \underbrace{r_{curiosity}(\pi)}_{\text{exploration reward}}$$

where:
- $\theta, \phi$ = encoder and adversarial masker parameters
- $\psi$ = RSSM/reward model parameters
- $\omega$ = Cross-JEPA critic parameters
- $\pi$ = curiosity exploration policy
- $\lambda_1 = 0.5, \lambda_2 = 0.3, \lambda_3 = 0.1$ (tunable weights)

The system minimizes $\mathcal{L}_{total}$ over recorded performances, then runs the curiosity loop and Dreamer online during real-time play. The adversarial masker and Cross-JEPA critic train offline.

**The result:** a system that plays, dreams about what it played, discovers what it can't predict, and trains itself to predict better — a perpetual self-improvement engine that never stops learning to listen.

---

*Document version: 1.0 | Project: fleet-jepa-midi | Date: 2026-08-13*
*Synthesized from research across DreamerV3, ICM, adversarial masking, Cross-JEPA, Procrustes alignment, and the JEPA literature. Ideation informed by ByteDance Seed-2.0-pro, NousResearch Hermes-3-Llama-405B, Qwen3-Coder-480B, NVIDIA Nemotron-3-Ultra, and ByteDance Seed-2.0-mini via DeepInfra.*
