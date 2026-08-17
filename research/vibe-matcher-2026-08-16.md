# Vibe Matcher — the Perception Agent for The Tap radio theater

**Date:** 2026-08-16
**Author:** fleet perception engineer (subagent)
**Status:** v1 shipped and run against the rendered corpus

---

## 1. The problem

The Tap renders *many separate* audio clips — different TTS voices (Flash, Pro,
Wesley, Hermes, Lucineer…), different moments, different energies. Played back
in filename order, the broadcast is a *playlist*. The captain's directive is to
make it feel like *one show*.

That requires an **ear** — a perception layer that listens to each rendered
recording, learns its *vibe*, and uses that to **order** the clips and **blend**
the transitions so the station is coherent, not shuffled.

This is exactly the job JEPA (Joint Embedding Predictive Architecture) is
designed for: *feel the abstract state of the signal, not its surface*. But the
fleet's JEPA repo (`fleet-jepa-midi`) currently only has a **MIDI** encoder
(16 hand-crafted features/bar → 256-dim projection, EMA + stop-gradient +
cosine objective). There is no audio encoder yet, and no trained weights.

So v1 ships the **same pattern** the repo already uses for MIDI, applied to
audio: a deterministic hand-crafted feature extractor ("the ear") + a
continuity objective that plays the role the JEPA predictor will one day play.

## 2. What was built

`vibe_matcher.py` (repo root) — a self-contained Python perception engine
(librosa + numpy, both already on the host). Three stages:

### 2.1 LISTEN — per-clip acoustic analysis → `clip-manifest.json`

For every rendered clip in a directory it extracts:

| Signal | What the ear hears |
|--------|--------------------|
| `duration_sec` | how long the moment is |
| `loudness` (mean/std/max RMS, dBFS) | overall level / presence |
| `energy_curve` (20 bins) | the *shape* of intensity over time |
| `head_energy` / `tail_energy` | energy at the **boundaries** (cross-fade signal) |
| `spectral.centroid` (mean/std) | brightness — treble vs. bass voice |
| `spectral.rolloff` | where the energy rolls off |
| `spectral.flatness` | tonal (music) vs. noisy (breath/fricative) |
| `spectral.tilt` | dB/decade slope — "dark" vs. "bright" voice |
| `zcr_mean` | zero-crossing rate — air/noisiness |
| `tempo_bpm` | pacing |
| `mfcc_mean` + `mfcc_std` (13+13) | timbre fingerprint |
| `kind` | naive speech-vs-music label |

### 2.2 FEEL — directional vibe-continuity

Between any ordered pair `A → B` it computes a weighted **vibe-continuity
score** in `[0,1]`. The weights (renormalised over whichever signals are
present):

- **boundary_energy (0.30)** — how close A's *tail* is to B's *head* (the
  cross-fade signal)
- **timbre (0.25)** — cosine similarity of the MFCC + spectral-shape vector
- **loudness (0.15)** — overall level delta
- **brightness (0.10)** — spectral-centroid delta
- **pace (0.10)** — tempo delta
- **semantic (0.10)** — cosine similarity of **local `nomic-embed-text`
  embeddings** of each clip's transcript (when a matching `.md` exists)

The score is **directional** because a broadcast flows A→B; A's tail must land
in B's head. Each transition records its *weakest* signal — what the ear
noticed most — as a human-readable reason.

### 2.3 ORDER — greedy nearest-neighbour → `order.json`

Starting from the most central clip (highest mean outgoing continuity), it
greedily appends the best-flowing unplaced clip, producing an optimal order and
a suggested **cross-fade ms** per transition:

- silence at either boundary → clean cut (`0 ms`)
- smoother transition → shorter fade
- jarring jump → longer blend (masks the seam)
- music-involved boundaries get a little extra room

## 3. Results on the rendered corpus

Corpus: `/home/eileen/projects/ai-writings/speeches/` (16 clips decoded; 10 got
semantic text embeddings from `nomic-embed-text`).

Optimal order (first five transitions):

| # | from → to | continuity | cross-fade | weakest signal |
|---|-----------|-----------:|-----------:|----------------|
| 1 | the-compile-silence-tts → the-compile-silence-v2-tts | 0.957 | 375 ms | pace |
| 2 | the-compile-silence-v2-tts → song-2-wire-tts | 0.939 | 800 ms | brightness |
| 3 | song-2-wire-tts → song-4-hull-tts | 0.915 | 850 ms | brightness |
| 4 | song-4-hull-tts → puffins-dont-quit-v2-tts | 0.877 | 925 ms | loudness |
| 5 | puffins-dont-quit-v2-tts → the-first-fold-tts | 0.896 | 500 ms | semantic |

Full sequence: *compile-silence → compile-silence-v2 → wire → hull →
puffins-v2 → first-fold → first-fold-v2 → puffins → tide → pfd → fathoms →
towfish → vhf-tts → vhf-gateway → underscore* — closing on the instrumental
underscore and the full `.wav` mix, a natural sign-off for a show.

Total continuity **0.849** across 15 transitions; average cross-fade **828 ms**.

Outputs were written to the corpus dir (`clip-manifest.json`, `order.json`).

## 4. Why this makes "many clips sound like one show"

A playlist is orderless and hard-cut. A *show* has a shape: it opens, it moves,
it breathes, it lands. The perception agent supplies the two things that create
that illusion without regenerating any audio:

1. **Continuity of feel.** By ordering on tail→head energy match, timbre
   similarity, and level/brightness/pace deltas, consecutive clips *agree* with
   each other. The listener's ear never hits a wall of loudness or a timbre
   whiplash. The station has a *vibe* that carries across speakers and moments.

2. **Blended seams.** The cross-fade is chosen per transition: silence gets a
   clean cut, a smooth handoff gets a short blend, a jarring one gets a longer
   overlap. The result is a continuous stream rather than N separate files.

The v1 ear is *hand-crafted* — it's the acoustic analogue of the repo's
16-feature MIDI `BarFeatures`. That is deliberate: it works today, it is
inspectable, and it defines the exact target a learned encoder must reproduce
(section 5).

## 5. Roadmap — the top 3 next steps toward a real JEPA audio agent

The end-state is a **learned audio-JEPA**: a context encoder that turns a
window of a rendered clip into a latent "vibe" embedding, an EMA target
encoder, and a predictor — trained so the predictor's guess of the *next*
window's embedding matches the real one, with VICReg anti-collapse. Then
ordering + cross-fade fall out of *embedding distance* instead of hand-tuned
deltas.

**Top 3 next steps:**

1. **Add an audio encoder to `fleet-jepa-midi` and pair it with the existing
   MIDI JEPA scaffolding.** The repo already has `src/jepa/{embedding,predictor}.rs`
   with EMA + stop-gradient + cosine objective, plus a harness. Add a
   mel-spectrogram → latent encoder (start small: log-mel → 2–4 conv/Conformer
   layers → 384-dim, mirroring the README's target) that produces *audio* bar
   embeddings alongside the existing *MIDI* `BarFeatures`. Reuse the exact same
   predictor and objective — that is the whole point of JEPA's separation of
   perception from generation. Keep the 16-feature extractor as the
   deterministic baseline to beat.

2. **Train it self-supervised on the rendered corpus (no labels).** The
   `speeches/` mp3/wav files are the dataset. Mask 50% of a window, predict the
   masked portion's *embedding* (not the waveform). This needs no transcript, no
   tags — just the audio. Track a `vibe-reconstruction` metric (cosine distance
   between predicted and EMA-target embeddings) and make sure the VICReg
   variance term keeps embeddings from collapsing. This is what turns
   `vibe_matcher.py`'s hand-crafted ear into a *learned* ear that can feel
   qualities (tension, warmth, humor, gravity) the 16 features can't name.

3. **Replace the hand-crafted continuity score with embedding distance, and
   feed it back into `fleet-radio`.** Once clip embeddings exist, `vibe_matcher`
   becomes thin: `continuity(A,B) = cosine(emb_A, emb_B)` (optionally
   directional: `emb` of A's last window vs B's first). Then wire it into the
   `fleet-radio` pipeline (currently in `src/pipeline.ts`) so the nightly
   broadcast is *automatically* ordered and cross-faded by the perception agent
   — "many clips, one show" as an always-on process, not a one-shot script.

Beyond the top 3: add a cross-fade *renderer* (ffmpeg `acrossfade` driven by
`order.json`) so the agent doesn't just suggest seams but produces the final
mix; and add a learned speech-vs-music classifier to replace the naive
filename heuristic in `kind`.

---

*"You know? You get it. You feel it."* — that's the whole spec. v1 gives the
station an ear. The next three steps give that ear a memory.
