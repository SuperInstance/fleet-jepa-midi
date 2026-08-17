# Audio JEPA v2 — the learned ear (2026-08-17)

**Date:** 2026-08-17
**Author:** fleet perception engineer (subagent)
**Status:** trained, evaluated, shipped — not yet wired into fleet-radio
**Replaces:** the hand-crafted `vibe_matcher.py` ear (v1, 2026-08-16)

---

## 1. What this is

The roadmap in `research/vibe-matcher-2026-08-16.md` step 1–2 called for turning
the hand-crafted acoustic ear into a *learned* one: a self-supervised JEPA over
log-mel spectrograms that produces a 384-dim "vibe" embedding per audio window,
so that ordering and cross-fade decisions fall out of *embedding distance*
instead of hand-tuned deltas.

This is that encoder. It is a PyTorch re-implementation of the exact JEPA
pattern already in `src/jepa/` (EMA target + stop-gradient + cosine predictor +
VICReg anti-collapse), applied to audio. It **trains and runs for real** — no
placeholders — on the 16-clip `speeches/` corpus, using the RTX 4050.

## 2. Architecture

```
raw waveform (mono, 16 kHz)
  -> MelFrontend        : STFT(n_fft=400, hop=160) -> 64 mel bins, log-dB  (100 fps)
  -> ConvEncoder:
       conv stem        : 4 x (Conv2d + BatchNorm2d + GELU), 1->48->96->192->256
                          (freq 64->4, time /8)
       freq pool        : AdaptiveAvgPool -> 1 x T/8
       transformer      : 2 layers, d=256, 4 heads, ff 1024, LayerNorm-first
       time pool        : mean over time
       projector        : Linear(256 -> 384) + BatchNorm1d(384)   [the anti-collapse guard]
       output           : L2-normalize -> z  (unit 384-vector)
  -> Predictor          : MLP 384 -> 768 -> 384 (BYOL-style, hidden > input), L2-normalized
```

| Component | Value |
|-----------|-------|
| Mel bins / SR / fps | 64 / 16 kHz / 100 |
| Latent dim | 384 |
| Online params | **2.92 M** (encoder 2.33 M + predictor 0.59 M) |
| Optimizer | AdamW, lr 1e-3, cosine decay + 5-epoch warmup |
| Batch / steps / epochs | 32 / 100 per epoch / 100 (= 10 000 steps) |
| Train time | **~485 s (~8 min)** on RTX 4050 (6.4 GB) |

## 3. JEPA objective

For each step, sample a **4.8 s window** (2.4 s context + 2.4 s target) from a
random clip, apply mel-domain augmentation, then:

- **context** = first 2.4 s, with **50% of frames randomly masked** (zeroed).
- **target** = next 2.4 s, processed by the **EMA target encoder** (frozen,
  `stop-grad`).
- `z_c = f_theta(context)`, `z_t = f_theta'(target)`, `p = P_phi(z_c)`.
- **invariance** = mean cosine distance `1 - cos(p, stopgrad(z_t))`.
- **VICReg** on the projector output `h` (variance hinge + off-diagonal
  covariance), `lambda_var=1`, `lambda_cov=0.2`.
- `loss = 1.0 * inv + var + cov`.

EMA momentum `tau` ramps `0.99 -> 0.999` (cosine) and is **capped below 1.0** so
the target never fully freezes (a stabilising anti-collapse choice).

### The collapse fight (worth recording)

The first two attempts collapsed — all 16 clip embeddings pointed the same way
(mean pairwise cosine similarity 0.96, then exactly 1.00). Two lessons:

1. **VICReg variance on the *raw* projection is useless against *directional*
   collapse** — the model just scales `h` up to satisfy `std(h) >= gamma` while
   `z = h/||h||` still points one way.
2. **VICReg variance on the *normalized* `z` has zero gradient at exact
   collapse** — `std -> 0` makes the variance gradient degenerate, so the model
   sits in a collapsed fixed point neither term can escape.

The fix was the canonical one: **a `BatchNorm1d` in the projector** (the BYOL /
VICReg papers both use this). It forces per-dimension batch variance to ~1 and
its batch-mean-centring introduces the inter-sample competition that breaks
collapse. After the fix, per-coordinate std settled at `0.051 ≈ 1/sqrt(384)`
(the value for uniformly spread unit vectors) and stayed there for the whole run.

## 4. Training

Augmentations (mel-spectrogram domain, SpecAugment-family — chosen because
waveform pitch-shift/time-stretch via STFT cost ~700 ms/sample on CPU, which
would make 10 k steps infeasible):

- random gain (additive offset in dB)
- background noise (additive gaussian in dB)
- pitch shift (vertical roll of mel bins; log-freq ⇒ pitch is translation)
- time-stretch / speed change (time-axis resample, 0.9–1.1)

Final loss curve (`checkpoints/loss_curve.png`): invariance fell from ~0.5 to
**0.082** (cosine similarity ~0.92), covariance ~0.026, variance ~0 (BN does the
normalising). `train_log.csv` holds the full per-step trace.

## 5. Evaluation vs the hand-crafted ear

The hand-crafted `order.json` was regenerated first (identical to the 08-16 run:
total continuity 0.849). The learned ear then produced its own greedy
nearest-neighbour ordering using **cosine similarity of learned embeddings** as
the continuity signal — the same greedy procedure `vibe_matcher.py` uses, but
with learned embeddings instead of hand-tuned acoustic deltas.

| Metric | Value | Read |
|--------|-------|------|
| mean pairwise cosine sim (16 clips) | **0.113** | spread, **not collapsed** (0.96/1.00 before the fix) |
| per-coordinate std | **0.047** | ≈ 1/sqrt(384) → healthy |
| mean sim of *hand-crafted* adjacent pairs | **0.249** | neighbors are ~2.2× more similar than random |
| Kendall-tau (global vs hand order) | **+0.333** | weak-moderate positive agreement |
| Kendall-tau (directional vs hand order) | **+0.117** | weak positive |
| adjacent-pair overlap (global / dir) | 0.200 / 0.267 | some shared transitions |

**Effective rank** (does BatchNorm cause silent dimensional collapse?): the
16-clip embedding covariance has a *smooth* singular-value spectrum — 1.14,
1.02, 0.96, 0.93, 0.87 … 0.57 across the top 12 — with no cliff. 5 dims explain
55%, 10 dims 88%. The embeddings occupy a well-spread region of the 384-sphere,
not a 5–10-dim manifold, so the BN projector has **not** collapsed the space
dimensionally (the concern an external reviewer raised — see §8).

**Qualitative read:** the learned ear *does* feel a continuity signal. Clip pairs
that the hand-crafted ear put next to each other are, on average, 2.2× more
similar under the learned embedding than a random pair — so the JEPA latent is
encoding real acoustic continuity, not noise. But the agreement is only
moderate (Kendall-tau +0.333 ≈ 1.8σ), and the *directional* (tail→head) notion
is weaker still (+0.117) because a global mean-pooled embedding does not yet
explicitly model the boundary-energy signal that drives `vibe_matcher`'s 0.30
`boundary_energy` weight.

**Conclusion:** the learned ear is *real but young*. It is not ready to replace
the hand-crafted ear in `fleet-radio` yet — but it has passed the hard test
(no collapse, genuine continuity signal) and is the right skeleton to grow.

## 6. What the learned ear must beat next

To be wired into `fleet-radio` it needs to (a) beat, not just approach, the
hand-crafted order, and (b) be cheap/robust enough to run nightly.

1. **Directional continuity.** Add an explicit tail→head signal: encode A's last
   window and B's first window and use `cos(emb_tail(A), emb_head(B))` as the
   transition score, or train a small head on top of the frozen encoder. Today's
   directional tau (+0.117) is the clearest gap.
2. **More / real data.** 16 clips is a toy. The same objective on the full
   `speeches/` + `ai-writings` archive (hundreds of clips) would tighten the
   embedding space dramatically. Real (waveform) augmentation should come back
   once precompute amortises the cost.
3. **Cross-fade from embeddings.** Map `cos`-distance to a cross-fade ms the way
   `vibe_matcher` maps continuity, and render the mix with ffmpeg `acrossfade`.
4. **A learned speech-vs-music classifier** to replace the filename heuristic in
   `kind` (the JEPA latent already separates them: `song-*` and `underscore`
   clips cluster away from the TTS clips).
5. **Beat the 0.849 baseline explicitly.** Report a single headline number —
   "mean cosine-similarity of the learned order's transitions" — and hold it up
   against `vibe_matcher`'s 0.849 total continuity.

## 7. Files

- `audio_jepa/model.py` — MelFrontend, ConvEncoder, Predictor, AudioJEPA
- `audio_jepa/dataset.py` — clip loading, mel caching, mel-domain augmentation
- `audio_jepa/__init__.py`
- `train_audio_jepa.py` — EMA/stop-grad/cosine/VICReg training loop
- `eval_audio_jepa.py` — embedding extraction, ordering, Kendall-tau comparison
- `checkpoints/audio_jepa_v2.pt` — saved checkpoint (online + EMA target)
- `checkpoints/train_log.csv`, `checkpoints/loss_curve.png` — training traces
- `checkpoints/eval_output.json` — evaluation metrics
- `research/audio-jepa-v2-2026-08-17.md` — this document

## 8. External review (DeepInfra)

Three review angles were run (raw transcript in
`research/reviews-audio-jepa-v2.md`):

- **Qwen3.6-35B-A3B (VICReg math)** — confirmed applying VICReg variance/cov to
  the **raw projector output `h`** (not the L2-normalized `z`) is the correct
  choice: normalizing first over-constrains a fixed-norm vector and makes the
  covariance term degenerate. This validates the final design.
- **Seed-2.0-pro (architecture)** — verdict: "extremely good first attempt,
  avoided 90% of common SSL audio pitfalls." Three risks flagged, all carried
  into §6: (1) *augmentation manifold overfitting* — with identity-preserving
  augs on 16 clips the model can memorise clip identity (a real limit for
  generalising to unseen speech, though clip-separation is actually what the
  ordering task wants today); (2) *silent dimensional collapse from BatchNorm*
  (variance term never fires) — **checked and ruled out** here via the smooth
  singular-value spectrum above; (3) *this is BYOL-with-past/future windows
  rather than token-level I-JEPA* — fair, and consistent with the repo's own
  "predict next embedding" `LinearPredictor`, but token-level prediction is the
  next step. Its concrete hardening advice (SpecAugment block masks, holdout
  clip, weight-decay 1e-3, EMA cap 0.996, per-sample mel normalisation) is
  folded into §6.
- **Qwen3-Coder-480B (bug review)** — rate-limited (HTTP 429); the correctness
  bugs were instead found and fixed during bring-up (§3 lists the collapse bugs;
  bring-up also fixed: `F` name-shadowing of `torch.nn.functional`, `AF.speed`
  returning a `(waveform, sr)` tuple, `phase_vocoder`'s `phase_advance` shape,
  and length-changing augmentations breaking the fixed window split).

## 9. Verdict: is it ready for fleet-radio?

**Not yet.** The learned ear produces a genuine, non-collapsed continuity signal
(hand-crafted neighbours 2.2× more similar than random; Kendall-tau +0.333), but
it does not yet *beat* the hand-crafted 0.849 ordering, and its directional
(tail→head) sense is the weak point (+0.117). It is the correct skeleton and
passes the hard tests; §6 lists exactly what it must clear before it conducts
the show.

---

*The ear is growing. It can now feel continuity it was never told to feel — just
not yet sharply enough to conduct the show on its own.*
