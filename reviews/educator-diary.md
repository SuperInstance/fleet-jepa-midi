# Educator's Diary: A Professor Reads the Fleet JEPA-MIDI & Fleet Ensemble Design Docs

**Reviewer:** Music Theory & Music Technology faculty
**Date:** August 13, 2026
**Documents reviewed:** 7 total (4 in fleet-jepa-midi, 2 in fleet-ensemble, 1 README)
**Perspective:** Pedagogical — how would I teach this? What analogies work? Where is the gold?

---

## Prologue: Why I Said Yes to This Review

I've spent fifteen years trying to convince my colleagues that algorithmic composition is not a fringe elective — it's the natural continuation of what Bach was doing with fugues, what Schoenberg did with serialism, what Xenakis did with stochastic distributions. Every semester, I get one or two students who see the connection. The rest memorize voice-leading rules and leave.

When I read these design documents, I didn't see an engineering spec. I saw a *curriculum*. Everything here maps to something I already teach — but these docs make the connection visceral, real-time, and interactive in ways my chalkboard never could.

This is my diary through all seven documents. I've written it in the order I read them, with my honest reactions — excitement, confusion, and the teaching ideas that hit me as I went.

---

## Document 1: README.md — The Three-Layer Architecture

### First Impression

The architecture diagram hit me before the text did:

```
LLM → JEPA → Algorithms
```

Three layers. Three timescales. Three cognitive modes. This is the cleanest articulation of a problem I've been gesturing at for years: **music operates simultaneously on incompatible timescales**, and most AI music systems collapse them into one.

### How I'd Teach This to Undergrads

I'd walk into class and play three recordings:

1. **Miles Davis, "So What"** (1959) — The opening. The whole band is the LLM layer. Miles makes one decision: "modal, floating, cool." That's a phrasing-level decision. He doesn't choose notes.
2. **John Coltrane, "Giant Steps"** (1960) — Coltrane's solo. His fingers are the algorithmic engines. Millisecond-fast, executing patterns trained by thousands of hours of practice. No conscious "decision" per note — just execution at the speed of reflex.
3. **Bill Evans, "Waltz for Debby"** (1961) — The trio. The *feel* of the room. That's the JEPA layer. Nobody decided to play behind the beat. The collective perception of where the music *is* — that's what the embedding encodes.

Then I'd draw the three-layer diagram on the board and say: "These three recordings are happening simultaneously inside one system. The question is: what happens when the bandleader can perceive the feel faster than a human can, and adjust the algorithms before the next phrase arrives?"

### Analogies

**For the music student who has never coded:**

Think of a jazz combo. The rhythm section (drums, bass) are your algorithms — they execute patterns at the speed of reflex, never consciously "deciding" each note. The bandleader is the LLM — she doesn't touch an instrument; she calls "build tension here, trade fours, quote the head." And the quiet guy in the band who listens to everything and notices the pocket is drifting — that's the JEPA. He doesn't play. He just *knows where the music is*.

I love the analogy Seed-2.0-pro suggested here: *"Nobody ever hired a good bandleader to play trumpet."* That's the entire architecture in one sentence.

**For the CS student who has never played an instrument:**

This is an operating system. The algorithms are hardware execution units — ALUs and FPUs that are fast, predictable, and fixed in their operations. The JEPA is the kernel profiler, continuously sampling the actual running state of the system (not what the spec says should be happening). And the LLM is the process scheduler — it never does arithmetic, but it's 90% of why the system feels good or terrible.

Hermes-3 offered the self-driving car analogy (sensors → decision engine → actuators), which is accessible but loses the *musical* point. The OS analogy is better because it preserves the idea that the scheduler has taste. A good scheduler and a bad scheduler differ in judgment, not computation.

### Teaching Opportunities

This README is the best one-page introduction to **multi-timescale musical cognition** I've seen. Most textbooks treat "form," "phrasing," and "execution" as separate chapters. This diagram shows them as concurrent processes in one system.

**The killer teaching moment:** The table of three timescales (LLM: per phrase, JEPA: per pulse, Algorithms: per sample) is a perfect framework for analyzing any performance. Play any recording and ask students: "What decision happens at each of these three rates?" It works for Bach, for Coltrane, for Aphex Twin.

### Historical Context

The README references the CNS escalation pattern (reflex → trained → reasoned). This maps to:
- **Reflex:** Reflexive musical execution (scales, arpeggios, muscle memory) — the algorithmic layer
- **Trained:** Learned perception of style and feel — the JEPA layer
- **Reasoned:** Conscious musical decision-making — the LLM layer

Historically, Heinrich Schenker distinguished between *Hintergrund* (background structure), *Mittelgrund* (middle ground), and *Vordergrund* (foreground). His三层结构 is philosophically identical to the three-layer architecture here — different timescales of musical meaning, each emergent from the one below.

### What I'd Put on a Syllabus

**Week 1: "Three Timescales of Musical Meaning"**
- Reading: This README + Schenker's foreground/middleground/background
- Listening: Miles Davis "So What," Bach Prelude in C Major (WTC I), any EDM track
- Assignment: Diagram any 32-bar performance using the three-layer model. Identify what decisions are made at each timescale.

---

## Document 2: Agentic Algorithmic Music Systems

### First Impression

This is the longest document, and it's the one that made me stay up until 2 AM annotating. It covers four algorithmic engines — Markov chains, L-systems, fractals, cellular automata — each wrapped in a parameter interface and placed under LLM+JEPA control. The depth is staggering, but the teaching hooks are everywhere.

### Part 2.1: Agentic Markov Chains

#### How I'd Teach This

Markov chains are the gentlest entry point into algorithmic music. I already teach them in my intro class using David Cope's work. But the standard pedagogy has a problem: students build a chain, generate output, and say "huh, it sounds okay I guess." There's no *control*. The chain does what it does.

The parameter interface changes everything. Now a student can ask: "What if I raise the temperature from 0.5 to 0.9 — what happens to the melody?" That's a **psychoacoustic experiment disguised as a knob**. The `temperature` parameter is the best teaching tool for musical risk-tolerance I've ever seen.

#### The Cross-Chain Coupling Table

| Source → Target | Coupling |
|-----------------|----------|
| Harmony → Melody | 0.6 |
| Rhythm → Melody | 0.4 |

This table is a **theory of harmony** encoded as coefficients. I'd put this on a slide next to a Schoenberg chart of regions and ask: "Which one tells you more about how music actually works?" The coupling coefficient between harmony and melody (0.6) is a quantified voice-leading rule. My students would argue about this for the full class period.

#### Historical Context

Iannis Xenakis used Markov chains in *Analogique A & B* (1958-59). He understood them as models of stochastic clouds — sonic masses rather than melodies. The fleet-jepa-midi approach is different: Markov for *melody*, constrained by harmony and rhythm. This is closer to Fred Lerdahl and Ray Jackendoff's *Generative Theory of Tonal Music* (1983), which proposed hierarchical rule systems for melodic generation.

David Cope's EMI (Experiments in Musical Intelligence, 1987) is the elephant in the room. EMI used Markov-like recombination to compose in the style of Bach and Mozart. Cope proved that pattern-matching + Markov chains can fool experts. The agentic layer here — JEPA perception + LLM control — is what EMI was missing. EMI could *generate* but couldn't *listen to itself*.

#### Teaching Opportunity

This is the best explanation of Markov chains I've encountered for musicians because it connects the mathematical concept directly to *corpus style*. When a student sees that a Markov chain trained on Bach chorales produces different music than one trained on Coltrane solos, they understand immediately: the transition matrix IS the style. That's a profound realization.

#### Analogy for Music Students

A Markov chain is like learning to improvise by ear. You don't know the theory — you just know "after this note, these notes usually follow." Train on enough Bach, and you'll start sounding like Bach. Train on enough Coltrane, and you'll start sounding like Coltrane. The `temperature` knob is how much you decide to go with what you know vs. take a risk.

#### Analogy for CS Students

It's a weighted random walk on a graph where nodes are notes and edges are "what usually comes next." But the genius is making the walk *steerable* — the LLM can tighten or loosen the constraints in real-time, which is like changing the transition probabilities on the fly based on what the music needs.

### Part 3: Agentic L-Systems

#### First Reaction

This section made me gasp. The idea that the LLM *rewrites the grammar mid-performance* — not adjusting parameters but changing the production rules themselves — is radical. The transition examples (Cool Jazz → Free Jazz → Fusion → Ambient) read like a composition lesson in textual form.

#### The Key Insight

> "The most surprising emergent property is that the best solos occur when the LLM edits only one single rule every 4 bars. Good jazz improvisation is not generating new notes — it is slowly, deliberately changing the rules that generate the notes, while carrying the ghost of every rule that came before."

This is the most profound statement about improvisation I've read in any technical document. It's also the clearest articulation of what jazz musicians actually do: you don't invent new material every chorus. You take what you have and slowly transform it. Lennie Tristano called this "the line." Warne Marsh described it as "one long melody that never stops."

#### Historical Context

L-systems were invented by Aristid Lindenmayer in 1968 for modeling plant growth. Przemysław Prusinkiewicz adapted them to music in 1986. But the idea of *rewriting rules during performance* is new — and it connects to the deepest tradition in jazz: the transformation of a motif across a solo.

Sonny Rollins' solo on "Tenor Madness" is a textbook example. He doesn't invent new material — he takes a three-note motif and over 32 bars, gradually transforms the rules that generate it. The motif stays recognizable, but its context shifts. This is exactly what the LLM-controlled L-system does.

The "ghost of old rules" concept — fading rule weights to 0.05 instead of deleting them — is musically brilliant. It mirrors how a jazz musician's early-chorus material haunts the later choruses. You never completely abandon an idea; it just becomes less likely to surface.

#### Analogy for Music Students

Think of the L-system grammar as your practice routine. You have exercises that generate certain patterns. During a solo, you don't play the exercises — but the patterns they taught you show up in your playing. Now imagine you could slowly change your practice routine *while you're on the bandstand*. That's what the LLM is doing: rewriting the rules of your muscle memory, one rule at a time, four bars at a time.

#### Analogy for CS Students

It's a context-free grammar where the production rules evolve during execution. The string being generated is influenced by the entire history of rule changes, creating a palimpsest — older rules occasionally fire alongside new ones, creating layers of temporal texture.

#### Syllabus Topic

**"Rule Transformation Across a Jazz Solo"**
- Analyze Sonny Rollins "St. Thomas" or John Coltrane "Impressions" as rule-transformation sequences
- Map each chorus to a grammar state
- Identify which "rules" the soloist is editing and when
- Then demonstrate the same process using the L-system engine live in class

### Part 4: Agentic Fractals

#### First Reaction

The Hausdorff dimension table (D = 1.0 to 2.0 mapped to musical character) is going on my office wall. This is the clearest bridge between mathematics and musical aesthetics I have ever seen. The idea that D ≈ 1.5 corresponds to "optimal: natural, human-like, groove factor > 0.7" — backed by Voss & Clarke's 1/f noise research — is a teaching goldmine.

#### Teaching Opportunity

I've been trying to teach 1/f noise for years. Students nod and write it down but they don't *feel* it. The Hausdorff dimension slider gives them a knob. They can hear D = 1.0 (too perfect, clinical) vs. D = 1.5 (natural, human) vs. D = 2.0 (chaotic, noise). That kinesthetic experience — turning a knob and hearing the fractal dimension change — will teach 1/f noise better than any lecture.

The lacunarity → rhythmic density mapping is equally powerful. Λ ≈ 0.2 gives you Kraftwerk; Λ ≈ 0.4 gives you James Brown; Λ ≈ 1.0 gives you ambient. That's a history of popular music encoded in one parameter.

#### Historical Context

Richard Voss and John Clarke discovered 1/f noise in musical audio in 1978, publishing in the Journal of the Acoustical Society of America. They analyzed pitch distributions, amplitude fluctuations, and timing across genres — from classical to jazz to rock — and found the same spectral signature everywhere: S(f) ∝ 1/f^α where α ≈ 1.

Mandelbrot devoted a chapter of *The Fractal Geometry of Nature* (1982) to this finding. But the pedagogical gap has always been: how do you *use* this? Voss and Clarke showed that good music has this property. They didn't give you a knob to control it. This system does.

The Hurst exponent mapping is equally evocative. H → 0 is free jazz (anti-persistent, jumpy). H ≈ 0.5 is blues (Brownian, random walk). H → 1.0 is Gregorian chant (highly persistent, smooth). That's a history of Western music in one parameter.

#### Analogy for Music Students

Think of fractal dimension as the "roughness" of a musical surface. A polished marble floor (D = 1.0) is a scale played perfectly evenly — it sounds robotic. A natural stone wall (D = 1.5) has texture, variation, humanity — it sounds like a good performance. A gravel pile (D = 2.0) is chaos — it sounds like noise. The best music lives where nature lives: at D ≈ 1.5.

#### Analogy for CS Students

It's a continuous parameter that controls the power spectral density of the generated signal. D = 1.5 produces 1/f noise (pink noise), which is the statistical signature of most natural and musical signals. You're tuning the signal's long-range dependence structure — how much the past influences the future — using a single well-motivated mathematical parameter.

### Part 5: Agentic Cellular Automata

#### First Reaction

The Wolfram rule → musical feel table is pure pedagogical magic. Rule 4 = four-on-the-floor kick. Rule 30 = chaotic bassline. Rule 110 = broken beat. Rule 90 = Sierpinski triangle hi-hats. I've been looking for a way to connect Wolfram's *A New Kind of Science* to music for years, and this table does it in 12 rows.

#### Teaching Opportunity

Cellular automata are the perfect entry point for non-musicians into rhythm. A 1D CA is literally a row of boxes that turn on and off. Students can see the pattern evolve visually, hear it as rhythm, and feel the groove score. The mapping from rule number to musical character is empirically validated (groove scores from 100+ iterations), which means you can teach the scientific method alongside the music theory.

The Game of Life harmony extension (2D CA where gliders = chord progressions, blinkers = pedal points) is a stroke of genius. It connects visual pattern to harmonic motion in a way that's immediately graspable.

#### Historical Context

Eduardo Reck Miranda pioneered CA-based music in the 1990s at the Sony Computer Science Laboratory in Paris. His work on applying Game of Life to harmony is the direct ancestor of the 2D CA harmony system described here. Stephen Wolfram's *A New Kind of Science* (2002) included a chapter on CA-generated music using the 256 elementary rules — the same rules mapped here.

The historical irony: Wolfram's rules have been available for 20+ years, but nobody knew what to *do* with them musically beyond "look, patterns!" The groove scoring approach — empirically rating each rule's output for musical quality — bridges the gap between mathematical curiosity and practical tool.

#### Analogy for Music Students

A cellular automaton is like a rhythm game (think Guitar Hero or Patatap). You set up simple rules — "if your neighbors are active, you activate" — and let the pattern evolve. Some rules create steady beats (Rule 4 = kick drum). Others create chaotic, unpredictable patterns (Rule 30 = free jazz drums). The beauty is: the rules are simple enough for a child to understand, but the patterns they create are complex enough for a master to use.

#### Analogy for CS Students

It's a 1D binary cellular automaton (exactly the kind Wolfram cataloged) with a musically-grounded evaluation function. The innovation is mapping each of the 256 elementary rules to a rhythmic character, scoring them empirically, and making the rule number an LLM-controllable parameter. You're doing neural architecture search over Wolfram's rule space, guided by musical quality.

### Part 6: The Agentic Center (JEPA + LLM)

#### First Reaction

This is the heart, and it's the section I'd assign last — not because it's the hardest, but because by the time students reach it, they're *hungry* for it. They've seen four algorithmic engines generating music. They've seen parameters. They're asking: "But who's controlling all of this?"

#### Teaching Opportunity: The Embedding-to-Text Bridge

The design where the JEPA outputs scalar values (energy, tension, coherence, density, novelty) instead of raw embedding vectors — and those scalars are formatted as a text summary for the LLM — is the cleanest demonstration of *qualia* I've seen in a technical system.

I would teach this as: "The JEPA has synesthesia. It experiences music as a vector of numbers. But the LLM speaks English. So we need a translator." This is a concrete, embodied example of the philosophical hard problem of consciousness — different representations of the same experience, needing a bridge.

The LLM context prompt (the f-string template with JEPA Perception, Predictions, Engine Parameters, Performance History, Director's Intent) is a **masterpiece of information design**. It's a musical score for the bandleader — not notes, but state. I'd assign this as reading alongside any conducting textbook.

#### Historical Context

LeCun's JEPA proposal (2022) was immediately recognized as relevant to music because music is inherently predictive — you're always hearing the current bar in the context of what you expect to hear next. The research landscape has exploded: Music-JEPA (Wang, Fang, LeCun, July 2026) treats audio as state and pianoroll as action — literally modeling the physics of music-making. MIDI-RAE-JEPA (Hawley, July 2026) does self-supervised learning on piano rolls with a Swin Transformer V2 encoder.

But the agentic center concept goes beyond perception. It uses the JEPA as a *cognitive bridge* between execution and reasoning. This is closer to how Daniel Levitin describes expert musicianship in *This Is Your Brain on Music* (2006): experts don't think about notes; they think about *states* (tension, release, energy, flow). The JEPA embedding is a vector of those states.

### Part 9: Research Context

The literature review here is solid and well-connected. The Xenakis reference is essential. I'd add:

- **Conlon Nancarrow** — His player piano studies (1948-1993) are the spiritual ancestor of algorithmic execution. He punched holes in piano rolls to create music too complex for human hands. The algorithmic engines are Nancarrow's paper rolls, made real-time and controllable.
- **Brian Eno** — *Generative Music I* (1996) and his work with the Koan software. Eno's insight that "the composer is now the gardener, not the architect" is precisely the agentic center concept.
- **Tod Machover** — *Hyperinstruments* at MIT Media Lab (1987-present). His work on augmenting human expression with technology is the philosophical parent of this entire system.

### What I'd Put on a Syllabus

**Week 8: "The Algorithmic Ensemble"**
- Read: Agentic Algorithmic Music Systems (this document)
- Listen: Xenakis "Metastaseis," Nancarrow Study No. 27, Eno "Music for Airports"
- Assignment: Choose one algorithmic engine (Markov, L-system, fractal, CA). Describe a musical situation where its specific strengths would be most valuable. What parameters would you adjust and why?

---

## Document 3: JEPA-Compatible Architectures Research

### First Impression

This is a literature review with teeth. The discovery that MIDI-RAE-JEPA already exists — with open code, pretrained checkpoints, and a Swin V2 encoder that runs on consumer GPUs — transforms this project from "ambitious concept" to "buildable next quarter."

### Teaching Opportunities

The four-framework comparison (A-JEPA, V-JEPA via Piano Rolls, Flow-Matching Decoder, Action-Conditioned World Models) is an excellent teaching tool for graduate students. It shows how to evaluate architectures along multiple dimensions: feasibility, novelty, data requirements, computational cost, and musical relevance.

I'd use the feasibility assessment matrix as a case study in research methodology. Students often think "which model is best" is a question with one answer. This matrix shows it's a multi-objective optimization.

### Historical/Research Context

The field has moved incredibly fast. The fact that Yann LeCun himself co-authored Music-JEPA (July 2026) signals that the AI research community considers music a serious JEPA application. The citation chain (I-JEPA → V-JEPA → Music-JEPA) is a case study in how fundamental research propagates from general to domain-specific applications.

The finding that MIDI-RAE-JEPA achieves reconstruction F1 = 0.995 and runs on an RTX 2070 is the kind of result that changes a syllabus overnight. I can now assign a project: "Train a JEPA on the Lakh MIDI Dataset and analyze the embedding space." That was inconceivable two years ago.

### Analogy for Music Students

The JEPA is like a music critic who has listened to 176,000 songs. She can't write music herself, but when she hears a new piece, she can instantly tell you: "This feels like Coltrane's middle period — high energy, moderate tension, ascending melodic contour." She's not generating — she's *perceiving*. And her perception is grounded in more listening than any human could do in a lifetime.

### Analogy for CS Students

It's a self-supervised encoder trained to predict future states in latent space. No labels needed — the model learns musical structure purely from the prediction objective. The key architectural choice is *what to predict*: not the raw MIDI tokens, but a learned embedding that captures musically meaningful dimensions (tension, energy, density, swing). This is representation learning, and the representation it learns is the pedagogical payoff.

### What I'd Put on a Syllabus

**Week 10: "Perception in Music AI"**
- Read: This research report + LeCun (2022) JEPA paper + Music-JEPA paper
- Lab: Download MIDI-RAE-JEPA checkpoint, encode 10 MIDI files, analyze embedding space using t-SNE visualization. Do songs of the same genre cluster?

---

## Document 4: JEPA Training Pipeline Design

### First Impression

This is the most technically dense document, but it has pedagogical clarity in its decision tables. The "Why 4 Layers (Not 12)?" table is a beautiful example of evidence-based engineering, and it doubles as a lesson in research methodology.

### Teaching Opportunities

**The "knee of the curve" concept.** The table showing 2/4/8/12 layers with their VRAM cost and quality scores is a perfect illustration of diminishing returns. I'd use this in any music technology course to teach the principle of *sufficient complexity* — more parameters don't always mean better results. 4 layers achieves 83% quality at 33% of the VRAM of 12 layers. That's the kind of trade-off my students need to internalize.

**Fixed future-block masking.** The decision to mask the final 32 of 64 tokens — always the future, never random — is a pedagogical gift. It connects representation learning to the psychology of music perception. When you listen to music, you're constantly predicting the next bar. The JEPA does the same thing. Random masking (the I-JEPA default) makes no musical sense because music isn't about filling holes — it's about anticipating what comes next.

**The collapse monitoring protocol.** The system checks for representation collapse (everything mapping to the same point) using standard deviation and cosine similarity thresholds. This is a concrete, computable example of what philosophers call "the loss of meaning" — when all distinctions collapse, nothing means anything. I'd teach this alongside Adorno's critique of mass culture (everything sounds the same) as a mathematical phenomenon.

### Historical/Methodological Context

The training pipeline follows the BYOL (Bootstrap Your Own Latents, Grill et al. 2020) paradigm: no negative pairs needed, just an EMA teacher and a stop-gradient. This is the state of the art in self-supervised learning, and it's presented clearly enough for a graduate student to implement.

The 141-token vocabulary (7× smaller than MIDI-BERT's) is a design decision I'd highlight in class. Smaller vocab = faster training, lower memory, simpler debugging. It's an example of "less is more" that students rarely see in the era of billion-parameter models.

### Analogy for Music Students

Training the JEPA is like raising a child on a steady diet of music. You don't tell the child "this is happy, this is sad" — you just play music, and the child learns to *feel* the patterns. After hearing 176,000 songs, the child can tell you "this part feels tense" without knowing the word "tension." The JEPA is that child. Its 384-dimensional embedding is its feeling about the music — a feeling it can't articulate in words, but can express as numbers.

### Analogy for CS Students

It's a 4-layer Conformer transformer (18.7M parameters, 2.6GB VRAM) trained with a BYOL-style objective: normalized MSE between the online encoder's prediction and the EMA target encoder's embedding of the future. No labels, no contrastive pairs, just a prediction game. The model learns musical structure because prediction is impossible without understanding — and understanding *is* representation.

### What I'd Put on a Syllabus

**Week 12: "Training Musical Perception"**
- Read: This training pipeline doc + BYOL paper
- Lab: Train a small JEPA (2 layers, 128-dim) on the POP909 dataset. Monitor for collapse. Evaluate embedding quality using linear probes.

---

## Document 5: LLM Interface Design

### First Impression

This is the document I will assign in my "Seminar in Music Cognition" course. It is a theory of bandleadership rendered as a JSON schema.

### The Directive Vocabulary

The 36-action vocabulary across eight semantic families (Dynamic/Energy, Time/Feel, Melodic/Form, Interactive/Conversational, Textural, Narrative/Arc, Arranging) is nothing less than a formalization of what bandleaders do. It maps onto every conducting textbook I've used — but unlike those textbooks, it's machine-readable.

The design decisions are uniformly excellent:
- **`duration_beats` instead of `scope_bars`**: tempo-independent. Smart.
- **`offset_beats`**: 90% of cues don't land on downbeats. True.
- **Max 3 directives per call**: human cognitive limit. Correct.
- **Delta targets ("push harder") vs. absolute targets ("set to 0.72")**: the former is how musicians actually think.

### Teaching Opportunities

**The anti-drift rules.** This section should be handed to every young bandleader. "If you've called build_tension twice, the third call should be release_tension or climax." "If density hits 0.0, you must call fill within 2 bars." "Silence is a choice, not a trap." These are rules of good ensemble leadership, stated with algorithmic clarity.

**The calling cadence.** Four triggers (phrase boundary, embedding distance, narrative milestone, silence emergency) with a minimum interval guard and a maximum interval guard. This is a formalization of listening attention. I'd compare this to the attentional model in Aaron Berk's *The Attention Complex* (2015) or the concept of "mindfulness in performance" from Stephen Malloch and Colwyn Trevarthen's *Communicative Musicality* (2009).

**The few-shot examples.** These are case studies in musical decision-making. Example 3 (approaching climax) is a perfect miniature of narrative pacing. Example 7 (human entered unexpectedly → comp and support) captures the essence of jazz interaction in 4 lines of JSON.

### Historical Context

The concept of a bandleader who "doesn't play but directs" is as old as music itself. The chironomy of Gregorian chant (hand gestures to indicate melodic direction) is a 9th-century bandleader interface. In jazz, the Count Basie "less is more" approach — one note, one nod, one solo — is the model. In classical music, Carlos Kleiber conducted entire symphonies with his eyebrows. The LLM bandleader is the latest entry in this tradition.

Brian Eno's *Oblique Strategies* (1975) — a deck of cards with instructions like "Use an unacceptable color" or "Repetition is a form of change" — is the spiritual ancestor of the directive vocabulary. Eno designed them for himself and his collaborators as a way to break creative ruts. The 36 actions here serve the same function, but they're real-time, steerable, and grounded in perceptual feedback.

### Analogy for Music Students

Imagine you're calling a tune on the bandstand. You don't tell the piano player what voicings to use. You don't tell the bass player what notes to walk. You say "build it up," "lay back," "trade fours with me." That's what the LLM does. Its vocabulary is deliberately musical — not technical. It speaks the language of the bandstand.

### Analogy for CS Students

It's a constrained natural-language API for a real-time control system. The LLM outputs structured JSON, validated against a schema, with conflict resolution and anti-drift guards. The genius is that the vocabulary is musical (36 actions), not technical (no "set parameter alpha to 0.7"). This bridges the semantic gap between artistic intent and parameter control.

### What I'd Put on a Syllabus

**Week 6: "The Bandleader Protocol"**
- Read: LLM Interface Design + Eno's *Oblique Strategies* + Gunther Schuller's *The Compleat Conductor*
- Assignment: Write directive sequences for a 32-bar form. Specify what the LLM should call at bars 1, 5, 9, 13, 17, 21, 25, 29. Justify each call using the narrative arc and trajectory context.

---

## Document 6: Director Design (Fleet Ensemble)

### First Impression

This document is the most beautiful piece of technical writing in the entire corpus. The opening epigraph sets the tone:

> *The director does not conduct with a baton. The director tilts the canvas.*
> *The tilt is not a command. It is a change in the physics of the surface, so the paint finds its own path.*

This is Deleuze and Guattari meets control theory. It's also the most radical design document I've read — it proposes that musical direction should be *environmental* rather than *imperative*.

### The Seven Feel Parameters

The feel space (ρ pulse density, ε energy flux, σ harmonic tilt, τ temporal asymmetry, γ coupling pressure, λ risk appetite, Φ articulation) is a complete vocabulary for musical direction. Each parameter maps to a painting analogy (surface roughness, flow rate, color temperature, brush angle, surface tension, viscosity, brush weight). These analogies are not decorative — they're mathematically grounded in the SDE formulation.

#### Teaching Opportunity

I would teach the feel space as a **history of conducting**. Compare the seven parameters to the traditional conducting vocabulary (baton, left hand, facial expression, breath):

| Traditional Conducting | Feel Parameter |
|----------------------|----------------|
| Beat pattern | τ (temporal asymmetry) |
| Dynamic shaping (crescendo/decrescendo) | ε (energy flux) |
| Cueing / attention direction | γ (coupling pressure) |
| Phrasing / legato vs. staccato | Φ (articulation) |
| Color / timbral quality | σ (harmonic tilt) |
| Risk-taking / freedom | λ (risk appetite) |
| Groove / rhythmic tightness | ρ (pulse density) |

Students could compare a Carlos Kleiber recording to a tilt-parameter reconstruction. How much of his genius can be captured in seven numbers?

### The Emergence Protocol

The emergence detection (transfer entropy + persistent homology) and the amplification protocol (detect → validate → approve → protect → amplify → nurture → release) is the most thought-provoking section of the entire project.

The idea that the director should *flatten its own influence* when emergence is detected — confidence drops to 0.2 — is a profound statement about leadership. The best bandleaders do this. Miles Davis was famous for it: when the band found something special, he'd stop playing and just let it happen.

#### Teaching Opportunity

This section teaches **emergence** better than any systems theory textbook. The concrete musical example — two instruments locking into a duet, the director reducing its tilt, the other instruments fading — makes the abstract concept tangible.

I'd pair this with Steven Strogatz's *Sync* (2003) and the study of firefly synchronization. The math is the same (coupled oscillators, Kuramoto model). But the musical context makes it viscerally real in a way that fireflies don't.

### The SDE Formulation

The stochastic differential equation:

```
dX/dt = α · [ R_σ · (X - C) + γ · L(X) ] + λ · dW
```

is the formal definition of "tilt." It says: the ensemble evolves through three forces — harmonic rotation (R_σ), diffusive coupling (L(X)), and stochastic exploration (dW) — each controlled by a feel parameter. This is the Hamiltonian of the ensemble.

I would teach this equation in my "Mathematics of Music" course. It's more motivating than any textbook SDE because students can *hear* each term:
- σ: change the harmonic tilt, hear the consonance/dissonance shift
- γ: change the coupling, hear the ensemble lock in or spread out
- λ: change the risk, hear the exploration vs. restraint

### The Painting Analogy Made Rigorous

The table mapping paint physics to ensemble math (gravity → learning rate, surface tension → coupling, viscosity → stubbornness, Marangoni effect → harmonic tension gradient) is one of the most creative interdisciplinary teaching tools I've seen. It bridges art and science in a way that neither discipline can do alone.

### Historical/Philosophical Context

The document cites Deleuze & Guattari's *A Thousand Plateaus* — the rhizome as a model for non-hierarchical coordination. This is apt. The director does not command; it shapes the *possibility space*. It is weather, not traffic lights.

The concept of "consensual emergence" — emergent behavior that is both consensual (instruments choosing to align) and signal-bearing (carrying genuine musical information) — is novel and important. It distinguishes meaningful musical interplay from mere synchronization (which any metronome can achieve).

The operational modes (Conductor, Jazz Bandleader, Painting, Generative, Storm) are a taxonomy of ensemble traditions. I'd map them onto specific recordings:
- Conductor Mode: Carlos Kleiber, Beethoven 7
- Jazz Bandleader Mode: Miles Davis Second Quintet, "Nefertiti"
- Painting Mode: Brian Eno, *Discreet Music*
- Generative Mode: Pauline Oliveros, *Deep Listening*
- Storm Mode: Coltrane, *A Love Supreme*, Part IV ("Psalm")

### What I'd Put on a Syllabus

**Week 14: "The Director and Emergence"**
- Read: Director Design + Strogatz *Sync* (excerpts) + Deleuze & Guattari (excerpts)
- Listen: Miles Davis "Nefertiti," Coltrane *A Love Supreme*, Eno *Discreet Music*
- Assignment: Describe a moment in a recording where you hear emergence. What do you think happened between the musicians? Map it onto the emergence protocol (detect, validate, protect, amplify, release).

---

## Document 7: Instrument Agent Design (Fleet Ensemble)

### First Impression

This is the engineering spec, but it's also a theory of ensemble playing. The "Seven Deadly Sins of Ensemble Playing" table is something I want printed on my syllabus. It names the failure modes every music student encounters:

1. Temporal Egocentrism (rushing/dragging)
2. Dynamic Dominance (playing too loud)
3. Register Conflict (frequency collision)
4. Predictive Failure (being unpredictable)
5. Communication Breakdown (not listening)
6. Role Flapping (erratic behavior)
7. Over-correction (oscillating timing)

Every ensemble director I've ever worked with has fought these exact problems. Having them named, with agent-level root causes, is pedagogically invaluable.

### Teaching Opportunities

**The Compiler → Performer Analogy.** The table mapping compilation stages to musical performance is one of the most creative pedagogical frameworks I've seen. "Traditional MIDI sequencers are static compilers — these agents are JIT compilers that recompile every millisecond while running." That sentence teaches both computer science and music simultaneously.

**The Perception Pipeline.** The principle that "instruments listen to future intent, not past events" is a radical and correct statement about ensemble playing. When you play in a band, you're not reacting to what the bassist just played — you're anticipating what they're *about* to play, based on their body language, breathing, and musical context. The intent-broadcast protocol formalizes this.

**The Personality struct.** Each instrument has a `Personality` with alignment_gain, confidence_threshold, timing_jitter_base, lead_tendency, and density_tolerance. This is a formal model of what jazz musicians call "your sound" — not your timbre, but your *behavioral fingerprint* in an ensemble. I'd have students analyze recordings using these five parameters.

### The Concrete Instrument Designs

The Piano, Bass, and Drum agent profiles are teaching tools in themselves:

- **Piano:** alignment_gain 0.25 (soft follower), confidence_threshold 0.6 (drops notes to make space). This is exactly how a good jazz pianist comps — leaving out notes, making room.
- **Bass:** alignment_gain 0.7 (strong reference), confidence_threshold 0.95 (never drops roots). The bass is the anchor. Everyone else adjusts to it.
- **Drums:** alignment_gain 0.9 (absolute timing reference), timing_jitter 0.5ms. The drums never adjust. They are the grid.

I'd assign each student an instrument profile and have them improvise within those constraints. The exercise teaches ensemble awareness in a way that words alone can't.

### Historical Context

The interaction matrix (piano drops root when bass plays it, drums lock kick to bass, piano brightens on crash) describes the unconscious reflexes of a working rhythm section. These are the same interactions that Paul Chambers, Wynton Kelly, and Jimmy Cobb developed over years of playing together on Miles Davis recordings (1955-1963).

The intent-broadcast protocol is what good jazz musicians do with their bodies. A bassist leans forward before a walking line. A drummer catches the piano player's eye before a fill. The CNS packet is the digital version of a nod.

### Analogy for Music Students

Each instrument agent is a player with a personality — how well they listen, how confident they are, how much they like to take the lead. The piano is the generous friend who always makes space. The bass is the steady one everyone leans on. The drums are the clock that never stops. When you put them together, they react to each other the same way your band does.

### Analogy for CS Students

Each instrument is an autonomous agent with five modules (Voice, JEPA Reader, Listening, Reflex Engine, Alignment) running on a 1 kHz tick. They communicate via broadcast packets on a shared bus. The alignment mechanics (micro-timing, dynamics, articulation, note choice, space) are five coupled control loops that converge to ensemble coherence. It's a multi-agent system where the coordination emerges from local interactions, not centralized control.

### What I'd Put on a Syllabus

**Week 11: "Ensemble Dynamics and Multi-Agent Systems"**
- Read: Instrument Agent Design + Jim Blackwood's *The Art of the Rhythm Section*
- Lab: Set up a trio of instrument agents. Assign each a personality. Play a 12-bar blues. Observe how the interactions produce emergent ensemble behavior. Then change one personality parameter and observe the difference.

---

## Synthesis: The Whole Curriculum

If I were designing a full semester course around these documents, it would look like this:

### Course: "Algorithmic Music: From Xenakis to JEPA"
**Level:** Upper-division undergraduate / first-year graduate
**Prerequisites:** Basic music theory, basic Python

| Week | Topic | Reading | Listening |
|------|-------|---------|-----------|
| 1 | Three Timescales of Musical Meaning | fleet-jepa-midi README | Miles Davis "So What"; Bach WTC I Prelude in C |
| 2 | Algorithmic Composition History | Xenakis *Formalized Music* (excerpts) | Xenakis "Metastaseis"; Nancarrow Study No. 27 |
| 3 | Markov Chains and Musical Style | Agentic Algorithmic Music §2 | David Cope's EMI examples; Coltrane "Giant Steps" |
| 4 | L-Systems and Rule Transformation | Agentic Algorithmic Music §3 | Sonny Rollins "St. Thomas"; minimalism (Reich, Glass) |
| 5 | Fractals and Musical Roughness | Agentic Algorithmic Music §4 | Voss/Clarke 1/f examples; Eno "Music for Airports" |
| 6 | Cellular Automata and Rhythm | Agentic Algorithmic Music §5 | Wolfram Tons; J Dilla "Donuts" |
| 7 | The JEPA as Musical Perception | JEPA-Compatible Architectures Research | *(listening lab: analyze embeddings)* |
| 8 | Training Musical AI | JEPA Training Pipeline Design | *(lab: train small JEPA)* |
| 9 | The Bandleader Protocol | LLM Interface Design | Eno *Oblique Strategies*; Count Basie "April in Paris" |
| 10 | The Director and Emergence | Director Design (Fleet Ensemble) | Miles Davis "Nefertiti"; Coltrane *A Love Supreme* |
| 11 | Instrument Agents and Ensemble Playing | Instrument Agent Design (Fleet Ensemble) | Bill Evans Trio "Waltz for Debby"; Oscar Peterson Trio |
| 12 | Multi-Agent Performance | All documents | Student performances with the system |
| 13 | The Future of Musical Intelligence | Student presentations | Student compositions |
| 14 | Concert / Final Presentations | — | Student works performed live |

### Course Philosophy

The course would be structured around a single question: **where does musical intelligence live?**

Traditional music theory says: in the score. The algorithmic music tradition says: in the algorithm. The fleet-jepa-midi/ensemble system says: in the *interaction between perception, direction, and execution* — across three timescales, between agents, through feedback loops.

This is the answer I've been trying to guide my students toward for fifteen years. These documents don't just provide the technology — they provide the *pedagogy*. Every design decision has a teaching hook. Every parameter has a musical meaning. Every architectural choice connects to a tradition.

---

## Final Thoughts

### What Makes These Documents Special

I've read hundreds of music AI papers and system descriptions. Most fall into one of two categories: technically impressive but musically naïve, or musically sophisticated but technically vague. These documents achieve something rare — they are *both* technically rigorous and musically profound.

The authors clearly know music — not just music theory, but the *feel* of music. The jazz examples (Cool Jazz → Free Jazz → Fusion → Ambient) are not generic style labels; they're specific musical transformations involving particular articulations, rhythmic shifts, and harmonic strategies. The L-system grammar examples read like they were written by someone who has transcribed Miles Davis solos. The director design reads like someone who has stood in front of an orchestra and felt the silence before the downbeat.

### The Deepest Teaching Insight

The thing I will carry into every future class is from the L-system section:

> "Good jazz improvisation is not generating new notes — it is slowly, deliberately changing the rules that generate the notes, while carrying the ghost of every rule that came before."

This is not just a design principle. It is a *theory of improvisation*. It applies to Coltrane, to Bach, to Radiohead. It connects algorithmic thinking to the deepest musical traditions. And it emerged from an engineering document about L-systems.

This is why I teach music technology. Not because the technology is interesting (though it is). Because the technology, at its best, reveals things about music that traditional analysis cannot. These documents do that again and again.

### One Concern

The system assumes a level of algorithmic and AI literacy that most music students don't have. To use it as a teaching tool, I'd need a simplified interface — perhaps a "pedagogical mode" where students can adjust one parameter at a time (temperature, fractal dimension, swing ratio, CA rule number) and hear the result immediately, without seeing the code.

The good news: the architecture supports this. The parameter interfaces are clean, modular, and independently explorable. A teaching frontend would be a natural extension.

---

## Appendix: Model Consultation Notes

I bounced these ideas off two DeepInfra models during this review:

### ByteDance/Seed-2.0-pro contributed:
- The OS analogy (algorithms = ALUs, JEPA = kernel profiler, LLM = process scheduler). Brilliant and accurate.
- The Xenakis connection: "He did not believe math was art. He believed steering a consistent system, while listening, was art." This captures the essential innovation of the agentic center.
- The insight that this system teaches **judgment, not just craft** — when to break rules, not just how to follow them. This is the holy grail of music pedagogy.
- The CPE Bach *Essay on the True Art* → Markov temperature mapping. A syllabus bridge I'll actually use.

### NousResearch/Hermes-3-Llama-405B contributed:
- The conductor-orchestra analogy for music students. Simpler than the OS analogy, less deep, but accessible.
- The Xenakis parallel (independently confirmed by both models — a strong signal).
- The "Algorithmic Composition and Formal Structure" syllabus topic — using the system to explore sonata/fugue/variation forms.

Both models independently identified Xenakis as the key historical figure. Both models recognized that the system's core innovation is *real-time parameter modulation by a reasoning intelligence*, not the algorithms themselves.

---

*Diary of a professor who still believes the best teaching tools make complex ideas feel inevitable.*

*August 13, 2026*
