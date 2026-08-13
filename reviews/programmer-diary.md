# Programmer's Diary — A Systems Programmer Reviews the Fleet

**Reviewer:** Senior Rust systems programmer (10 years real-time audio DSP + distributed systems)
** Date:** 2026-08-13
** Scope:** fleet-jepa-midi, fleet-ensemble, fleet-gateway, fleet-memory
** Verification:** Cross-checked key findings against DeepSeek (DeepSeek-chat, temp 0.7)

---

## Preface: What I Expected vs. What I Found

I was told this is a "three-layer real-time music intelligence system" with JEPA perception, algorithmic engines, and LLM direction. I expected to find:

- Actual MIDI I/O (midir, JACK, or at least PortAudio bindings)
- Lock-free ring buffers for the audio thread
- A real-time clock domain separate from the control plane
- Deterministic memory allocation in the hot path

What I found instead is a well-organized research prototype with strong architectural thinking but zero connection to physical audio. That's fine for a v1 — but the code comments sometimes make promises the system can't keep. Let me be specific.

---

## Part 1: fleet-jepa-midi (2,709 lines across 15 source files)

### Cargo.toml — Dependencies

```toml
midly = "0.5"        # SMF parsing only — not live MIDI
ndarray = "0.16"     # Unused in the hot path — where are the matrix ops?
clap = { version = "4", features = ["derive"] }
reqwest = { ... "rustls-tls", "json" }  # For gateway client
```

**Observation:** `ndarray` is declared but I see no `use ndarray` anywhere in the source. Dead dependency. The predictor (`jepa/predictor.rs`) uses raw `[[f32; 16]; 16]` arrays — which is fine for D=16 but means `ndarray` is paying compile-time cost for nothing.

**Risk:** Low. Just remove it or start using it.

**Missing:** No `midir`, no `cpal`, no `jack`, no `aubio`. This crate cannot receive or send a single live MIDI message. It parses `.mid` files and generates note sequences in memory. The `midi/stream.rs` `MidiStream` struct is a `VecDeque<(f32, MidiNote)>` — a logical buffer, not a hardware bridge.

### Module: `jepa/embedding.rs` (478 lines)

This is the strongest module. The 16 hand-crafted features are musically intelligent — harmonic tension via circular variance on the pitch-class wheel (lines 188-210) is genuinely clever. Whoever wrote this understands music theory.

**Latency profile:** `extract_features()` iterates over all notes in a bar multiple times (once for pitch stats, once for onset sorting, once for slice-based features). For a dense bar with 32 notes, that's ~5 passes over the data. At 125ms pulse rate with typical 4-16 note bars, this is sub-microsecond. **No latency concern.**

**Bug — double-counting bias in `predict()` (predictor.rs, line 27):**

```rust
pub fn predict(&self, current: &Embedding) -> Embedding {
    let mut out = self.bias;       // ← starts with bias
    for i in 0..EMBEDDING_DIM {
        let mut sum = self.bias[i]; // ← ALSO starts with bias
        for j in 0..EMBEDDING_DIM {
            sum += self.weights[i][j] * current[j];
        }
        out[i] = sum;
    }
    out
}
```

`out` is initialized to `self.bias`, then `sum` is ALSO initialized to `self.bias[i]`. The final assignment `out[i] = sum` overwrites the initial bias with the same value plus the weighted sum. So the result is correct, but the `let mut out = self.bias` line is dead code — `out` is fully overwritten in the loop. Minor, but it confused me for 30 seconds.

**The smoothing EMA (line 149):** α=0.12 → ~880ms time constant at 125ms pulse rate. This is reasonable for slow-moving spectral features but too slow for transient detection. A crescendo or dynamic shift will take ~3-4 seconds to fully register in the smoothed embedding. For a jazz bandleader calling "build energy," that's borderline acceptable. For detecting a surprise, it's too slow.

**Recommendation:** Dual-rate smoothing — α_fast=0.5 for velocity/energy features, α_slow=0.12 for register/tension features.

### Module: `jepa/predictor.rs` (132 lines)

The linear predictor is a 16×16 matrix multiply plus bias — 256 FLOPs per prediction. Trivially fast. The online update rule (line 62) is crude outer-product gradient descent:

```rust
self.weights[i][j] += lr * error[i] * current[j];
```

This is LMS adaptation. It works, but the learning rate is fixed and there's no normalization. If bars vary wildly in magnitude (e.g., silent → orchestral hit), the weights will blow up. I verified: the curiosity harness (`harness/curiosity.rs`, line 72) uses `lr = 0.01`, which is stable for the current feature normalization range (~[0,1]), but there's no guard against divergence.

**Will this meet a 10ms deadline?** Yes, easily. The entire predict+update cycle is nanoseconds.

**Will it meet a 10ms deadline at D=384 (the planned v2)?** 384² = 147,456 FLOPs per prediction. At modern CPU speeds (~10 GFLOPS scalar), that's ~15μs. Still trivially within budget. The concern at D=384 is memory bandwidth — the weight matrix is 588KB, which blows L2 cache. Expect 2-5× slowdown from cache misses. Still under 100μs.

### Module: `engine/` (ca.rs, markov.rs, fractal.rs, lsystem.rs)

These are offline generators. They produce `Vec<MidiNote>` with no timing guarantees. The Markov chain (`markov.rs`, line 65) picks the MIDDLE element of the transition list as its "random" choice:

```rust
let idx = next_intervals.len() / 2;
```

This is deterministic, not random. It means the Markov chain is actually a deterministic walk. The comment says "deterministic for reproducibility" but that's a design choice that eliminates the statistical properties that make Markov chains interesting musically. You've built a lookup table, not a stochastic process.

The fractal generator (`fractal.rs`) uses f64 escape-time computation — correctly avoids f32 precision issues in the Mandelbrot iteration. Good instinct. But `generate_params()` computes escape time for every note independently with no memoization. For 32 notes, that's 32 × up-to-64 iterations of f64 multiply-add. Still sub-millisecond, but wasteful if you're regenerating the same landscape.

The CA engine (`ca.rs`) is clean. Wolfram rule 30, toroidal wrapping, 16 cells. No issues.

The L-system (`lsystem.rs`) uses `HashMap<char, Vec<char>>` for rules — fine for v1 but a `Box<dyn Fn(&char) -> &[char]>` or even a match statement would avoid the hash lookup per symbol per iteration. At 3 iterations of a short axiom, this is irrelevant. At 10+ iterations, the exponential blowup makes the HashMap cost irrelevant anyway.

### Module: `midi/stream.rs` — `MidiStream`

The `VecDeque` with time-window pruning is a standard pattern. `prune()` (line 31) pops from the front while events are older than the cutoff — O(k) where k is the number of expired events. In the worst case (long silence followed by a burst), you get O(n) on the first insert after the gap. For a 2-second window at typical MIDI density (~50 events/sec), the buffer holds ~100 events. This is a non-issue.

**What's missing:** No capacity limit. A malicious or buggy producer could grow this unboundedly. Add a `max_capacity` with drop-oldest semantics.

**What's also missing:** This stream has no connection to any MIDI backend. It's purely logical. The `ingest()` method takes a `MidiNote` — who calls it? Nobody, currently. It's API-ready but unconnected.

### Module: `director/gateway.rs` — HTTP Client to fleet-gateway

```rust
let client = Client::builder()
    .timeout(Duration::from_millis(1500)) // 1500ms max per design doc
    .build()
```

**This is the latency bottleneck.** The Director calls `request_directive()` which does an HTTP POST to fleet-gateway, which then proxies to an LLM (OpenAI, DeepSeek, etc.). The 1500ms timeout is appropriate for LLM inference, but it means the Director's decision loop has a 1.5-second worst-case latency.

At 120 BPM (500ms per beat), a 1.5-second timeout = 3 beats of silence if the LLM is slow. **This cannot meet a 10ms deadline, and it shouldn't try to.** The design is correct: LLM direction is a slow control loop (1-5 seconds), not a real-time loop. But this must be explicitly decoupled — the real-time layer should never block on the Director.

### Module: `director/phrasing.rs` — Directive Vocabulary

42 actions across 8 families. Well-structured serde with snake_case rename. The `PhrasingCall` struct supports both absolute and relative scalar targets. This is good API design.

**Issue:** `DirectiveAction` derives `Hash` but also has `Custom(u8)` in `VoiceClass`-adjacent types that don't. If you ever want to put directives in a `HashSet` for deduplication, the current derive is correct. No issue here.

**Missing:** No validation that `intensity` is in [0, 1] or that `duration_beats` is non-zero. The LLM could send `intensity: -5.0` and it would serialize fine.

### Module: `harness/curiosity.rs` — Self-Improving Loop

The harness cycles through engines, generates 4-bar phrases, encodes them, and runs the LMS predictor. The curiosity reward IS the prediction error — high error means novel music.

**Architectural concern:** The harness uses the predictor's error as the reward signal AND updates the predictor on each observation (line 89). This creates a feedback loop: the predictor learns to predict the engines' output, error drops, "curiosity" decreases, and the harness concludes the system is no longer novel. But the engines haven't changed — the predictor just caught up.

This isn't a bug, it's a feature of intrinsic curiosity architectures. But the harness has no mechanism to then SEEK novel parameter regions. It just cycles through the same 4 engines with the same parameters. The `best_score` will monotonically decrease as the predictor learns, which looks like "the system is getting less creative" when actually it's "the model is getting more accurate." The v2 CMA-ES parameter mutation would fix this, but v1 is just a downhill run.

---

## Part 2: fleet-ensemble (3,207 lines across 14 source files)

### Cargo.toml — Edition 2024

```toml
edition = "2024"
```

This is the Rust 2024 edition. Good — `let` chains and other modern features available.

### The EMBEDDING_DIM Crisis

**This is the most critical finding in the entire review.**

- `fleet-jepa-midi/src/jepa/embedding.rs:8`: `pub const EMBEDDING_DIM: usize = 16;`
- `fleet-ensemble/src/protocol/mod.rs:9`: `pub const EMBEDDING_DIM: usize = 256;`

These two crates are supposed to form a pipeline: jepa-midi extracts embeddings, ensemble consumes them. But they speak different vector languages. A 16-dim vector from jepa-midi is not compatible with a 256-dim slot in fleet-ensemble's `CnsPacket::EmbeddingBroadcast`.

The comment in ensemble's `protocol/mod.rs` says "Start with 256; will grow when the real JEPA encoder is trained." This implies jepa-midi's 16-dim features are the "v1 placeholder" and ensemble is built for the "v2 real model." But nobody bridged the gap.

**Impact:** If you wire these two crates together today, you'll get a runtime panic or silent corruption (depending on how the data flows). `ListeningState::update_peer_embedding()` does `embedding[..n].copy_from_slice(...)` with `n = min(emb.len(), EMBEDDING_DIM)` — so a 16-dim embedding gets zero-padded to 256 dimensions. Zero-padding a 16-dim vector to 256 is not a meaningful embedding; it's garbage.

**Fix needed:** Either unify EMBEDDING_DIM across crates, or add an explicit projection layer (linear or learned) that maps 16→256.

### Architecture: CNS Bus

`tokio::sync::broadcast::channel::<CnsPacket>(1024)` — one broadcast channel for all packet types.

**Priority inversion risk (confirmed):** `FeelTilt` (Director, every 125ms, timing-critical) shares the same FIFO channel as `AgentPlayed` (event-driven, potentially bursty). A piano chord = 10 simultaneous `AgentPlayed` packets. If the channel is congested, the Director's pulse gets delayed behind note events.

At the current scale (3 instruments, 8Hz), the channel buffer (1024 capacity) will never fill. But the architectural choice is wrong for a real-time system. Separate channels for control (FeelTilt, DirectorParams) and data (AgentPlayed, EmbeddingBroadcast) would eliminate this class of problem.

**Vec<f32> in every CnsPacket:** The `EmbeddingBroadcast` variant contains `embedding: Vec<f32>`. Every broadcast clones the Vec for each subscriber. With 256-dim embeddings × 3 subscribers × 8Hz = 6,144 floats/sec cloned = ~24KB/sec of heap traffic. Negligible at current scale, but architecturally wasteful. `Arc<[f32; EMBEDDING_DIM]>` or a fixed-size array in the enum variant would eliminate per-message allocation.

**serde_json for packet encoding (packets.rs, line 290):**

```rust
pub fn encode_packet(pkt: &CnsPacket) -> Vec<u8> {
    serde_json::to_vec(pkt).unwrap_or_default()
}
```

JSON serialization on every packet. Not a latency problem (2-5μs per packet), but the wrong tool. `bincode` or `postcard` would be 5-10× faster, deterministic across platforms, and produce smaller wire sizes. The `unwrap_or_default()` silently swallows serialization errors and returns an empty vector — apacket that will fail to decode on the other end.

### Module: `director/perception.rs` — 5-Level Perceptual Stack

Well-designed. The centroid + dispersion + velocity + rotational flux + coherence model is sound for ensemble perception.

**`compute_coherence()` (line 153):** Uses `1.0 / (1.0 + variance)` on centroid magnitude time series. This is not actually Fourier stability — the comment says "Fourier stability" but the implementation is inverse-variance. It's a reasonable proxy, but mislabeled.

**`centroid_history` uses `Vec::remove(0)` for ring-buffer behavior (line 134):**

```rust
self.centroid_history.push(new_centroid);
if self.centroid_history.len() > HISTORY_WINDOW {
    self.centroid_history.remove(0);
}
```

`Vec::remove(0)` is O(n) — it shifts all elements. With HISTORY_WINDOW=32 and EMBEDDING_DIM=256, that's shifting 32×256×4 = 32KB per pulse. At 8Hz, that's 256KB/sec of memmove. Not catastrophic, but `VecDeque` with wrapping is the standard fix. Or just use `heapless::Deque<[f32; 256], 32>` for stack allocation.

### Module: `director/feel_space.rs` — 7-Dimensional Feel Manifold

The mathematical model is elegant. The `smooth_toward()` implementation (in `packets.rs`, line 159) does per-parameter exponential smoothing with clamping after. This is correct and stable.

**One concern:** `FeelSpace::space()` (feel_space.rs, line 21) computes `rho * lambda_inv` where `lambda_inv = 1.0 / lambda`. With `lambda = 0.01`, this gives `lambda_inv = 100`, and `space = rho * 100` clamped to 10. This means the "space" derivative parameter is extremely sensitive to small lambda values. A director mode with `lambda = 0.0` (full restraint) would cause division by zero — but the code guards with `if self.lambda > 0.01`. Good defensive coding.

### Module: `director/emergence.rs` — Transfer Entropy

The transfer entropy implementation (line 20) is a crude Granger-style approximation, not actual TE. The comment acknowledges this. The cross-correlation of A's magnitude with B's delta is a reasonable first-order proxy.

**But:** `detect_te_spikes()` (line 55) uses a hardcoded threshold of 0.5. This threshold has no statistical basis — it's not calibrated against a null hypothesis. In a real system, you'd want a permutation test or at least a rolling baseline. The current code will produce false positives whenever instruments are active.

### Module: `instrument/alignment.rs` — Spring-Damper + Kalman

This is where I found the most concerning issue.

**Spring-damper numerical instability (lines 78-95):**

```rust
let force = -k * x - c * v * 0.01;
self.timing_velocity += force;
self.timing_offset_us += self.timing_velocity * 0.001;
```

Problems:
1. **dt is implicit and inconsistent.** The velocity update uses `dt=1` (force is directly added). The position update uses `dt=0.001`. These are different time steps in the same integration — numerically invalid.
2. **The 0.01 damping factor is unitless magic.** Damping coefficient `c = 2 * sqrt(k * 1000)` (from `damping_coefficient()`), then multiplied by 0.01 in the force equation. This means effective damping = `2 * sqrt(k * 1000) * 0.01 ≈ 0.632 * sqrt(k)`. For k=0.075 (piano), that's effective damping of 0.173. The system is underdamped — it WILL oscillate.
3. **Euler integration with varying dt.** Tokio's `interval` is not hard-real-time. Under load, ticks drift. The Euler integrator doesn't account for this — it assumes dt=1 tick every time. This means the spring constant and damping ratio change with system load.

**Verified with DeepSeek:** This is mathematically unsound, not just inelegant. The system can oscillate or diverge under load. The fix is to measure actual dt between ticks and use it in the integration, or switch to a fixed-timestep lockstep model.

**Kalman filter (line 105):** Standard 1D Kalman with scalar gain. The Q (process noise) = 0.01 and R (measurement noise) = 0.1 are reasonable defaults for phase tracking. No issues here.

**`effective_offset_us()` (line 137):** The "don't correct below 5ms" humanization rule is musically intelligent. Real drummers don't correct sub-5ms deviations — that's feel, not error.

### Module: `instrument/reflex.rs` — Fast-Path Musical Responses

Clean implementation. The reflex engine evaluates peer events and returns `Vec<ReflexResponse>` — but nobody actually ACTS on these responses yet. The `InstrumentAgent::handle_packet()` calls `voice_class.check_reflex()` and logs the result (`debug!("Agent {} reflex: {:?}", ...)`) but doesn't trigger any MIDI output.

This is the right design for a stub: the reflex infrastructure exists, but the output path isn't wired. The <10ms claim is currently aspirational — there's no output path to measure.

### Module: `instrument/listening.rs` — Attention Allocation

Simple model: base attention by role, boosted by prediction error. The `update_attention()` recomputes all weights on every peer embedding update — O(n_peers²) per pulse if all peers update simultaneously. With MAX_INSTRUMENTS=32, that's 1024 operations per pulse. Trivially fast.

**Bug:** `update_peer_error()` (line 56) directly sets attention to 1.0 for high-error peers, bypassing the `update_attention()` recalculation. This means error-based attention is sticky — it stays at 1.0 until the next embedding update triggers a full recomputation. If an instrument's error drops to zero but it doesn't broadcast a new embedding, its attention weight stays pinned at 1.0.

### Module: `instrument/jepa_reader.rs` — Perception Pipeline

The JEPA reader uses **cosine distance** for prediction error (line 112), while `fleet-jepa-midi`'s predictor uses **L2 norm** (predictor.rs, line 53). These are different metrics. Cosine distance ignores magnitude; L2 doesn't. A shift from "all quiet" to "all loud" is a big L2 distance but zero cosine distance (same direction in feature space).

This metric inconsistency means the curiosity harness (L2) and the ensemble (cosine) will disagree on what's "surprising." Neither is wrong, but they should be consistent.

---

## Part 3: fleet-gateway — LLM Proxy with Circuit Breaker

### Architecture Assessment

This is the most production-ready crate in the fleet. It's a standard reverse proxy with:

- Provider chain with fallback (proxy.rs)
- Circuit breaker per provider (circuit_breaker.rs)
- Key rotation with bad-key tracking (key_chain.rs)
- Metrics and health endpoint (metrics.rs)
- Streaming responses (no buffering — O(1) memory per request)

**Well-engineered.** The circuit breaker state machine (Closed → Open → HalfOpen → Closed) is textbook correct. The key chain rotation with auto-reset when all keys are bad is sensible.

### Latency Analysis

The gateway sits between the Director and the LLM. Its job is to add as little latency as possible.

**Request path:**
1. Parse model from body (`extract_model` — serde_json parse, ~5μs)
2. Walk provider chain, check circuit breaker (Mutex<BreakerState>, ~1μs per provider)
3. Clone body bytes (`body.clone()` — memcpy of request size, ~1-10KB)
4. Forward to upstream provider (network latency: 100-2000ms)
5. Stream response back (no buffering)

**Total overhead: <50μs.** The gateway itself is not a latency bottleneck. The LLM is.

**Concern:** The `body.clone()` in the retry loop (proxy.rs, line 46) clones the entire request body for each attempt. For a large request (10KB), that's fine. For a request with a long conversation history (100KB+), repeated cloning on retries adds up. Consider `Bytes::clone()` which is cheap (Arc refcount), not `Vec<u8>::clone()`.

Actually, looking again: the body type is `bytes::Bytes` which IS Arc-backed. `Bytes::clone()` is O(1). Good choice. No issue.

### Circuit Breaker — Mutex Contention

Each provider has its own `CircuitBreaker` with internal `Mutex<BreakerState>`. In the hot path:
1. `allow_request()` — locks state, locks opened_at (2 Mutex acquisitions)
2. On success: `record_success()` — locks state, locks failures, locks successes, locks opened_at (4 Mutex acquisitions)

With 3-5 providers and sequential chain walk, that's 12-20 Mutex acquisitions per request. Under high concurrency (100+ concurrent requests), this becomes a contention point. But at fleet scale (a handful of LLM calls per second), it's fine.

**Risk:** If the gateway is shared across many agents (which is the plan), contention will increase linearly. Consider `AtomicBool` for the breaker state and `AtomicU32` for counters.

### Jemalloc

```rust
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;
```

Good choice for a server that handles many variable-size allocations (JSON bodies). Jemalloc's arena-based allocation reduces fragmentation vs. glibc's ptmalloc.

---

## Part 4: fleet-memory — Vector Index

### Architecture Assessment

SQLite + sqlite-vec with WAL mode and flock-based locking. Provider-tagged index files with atomic symlink swap. Streaming reindex with checkpointing.

This is a well-designed offline indexing system. The streaming approach (batch → embed → insert → checkpoint) is O(batch) memory, never O(corpus). This directly follows the critical path rule from the infrastructure proposal.

**Best crate in the fleet for production readiness.**

### `embedding_to_bytes()` Duplication

The function appears in both `db.rs` (line 295) and `search.rs` (line 71):

```rust
fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(embedding.len() * 4);
    for &v in embedding {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}
```

Should be `pub(crate)` in one place. Minor DRY violation.

### `Vec::remove(0)` in IndexManager

Same pattern as perception.rs — `Vec` used as a ring buffer with O(n) front removal. In the indexer, this doesn't matter (offline batch processing), but it's a consistency issue.

### `flock` via `extern "C"`

```rust
extern "C" {
    fn flock(fd: c_int, operation: c_int) -> c_int;
}
```

Direct FFI to `flock(2)`. This is correct on Linux but should use `libc::flock` from the `libc` crate for portability. On macOS, `flock` has slightly different semantics with NFS locks. For a WSL2-targeted system, this is fine.

### SQLite WAL Mode

```rust
conn.pragma_update(None, "journal_mode", "WAL")?;
conn.pragma_update(None, "synchronous", "NORMAL")?;
```

WAL + NORMAL synchronous is the right tradeoff for an index that can be rebuilt. Crash-safe enough — WAL survives power loss, synchronous=NORMAL means at most one transaction of data loss on kernel panic. For a searchable code index, that's acceptable.

**But:** `open_readonly()` also sets `journal_mode = "WAL"`. Setting journal mode on a read-only connection is a no-op (SQLite ignores it), but it's misleading. Remove it from the readonly path.

---

## Part 5: Cross-Cutting Concerns

### The 10ms Deadline Question

**Can this system meet a 10ms real-time deadline?**

No. And it shouldn't try to. Here's why:

The system has three latency tiers:
1. **Reflex layer (<10ms target):** `reflex.rs` evaluates peer events in microseconds. But there's no output path — no MIDI I/O, no audio callback. The infrastructure exists but can't act.
2. **Perception loop (125ms target):** JEPA encoding + alignment + feel-space smoothing. Currently runs in <1ms. Meets deadline easily.
3. **Director loop (1-5s target):** LLM direction via fleet-gateway. Cannot and should not meet 10ms.

The design correctly separates these tiers. The problem is that the reflex layer has no hardware connection, making the <10ms claim technically true (the computation is fast) but practically meaningless (there's nothing to output to).

### Dependency Risk Assessment

| Dependency | Risk | Reason |
|-----------|------|--------|
| `tokio` (full) | **Medium** | Pulls in everything including `tokio::fs`, `tokio::net`, etc. For a real-time system, `features = ["rt", "time", "sync"]` would be leaner. |
| `reqwest` 0.12 | **Low** | Well-maintained, uses hyper 1.0. rustls-tls avoids OpenSSL. |
| `midly` 0.5 | **Medium** | Last release 2023. SMF only. Need `midir` for live MIDI. |
| `ndarray` 0.16 | **Low** (unused) | Remove or start using. |
| `dashmap` 6 | **Low** | Excellent concurrent map. No issues. |
| `axum` 0.8 | **Low** | Solid web framework. Appropriate for gateway. |
| `serde_json` | **Medium** | Used for real-time packet encoding. Should be `bincode` or `postcard`. |
| `sqlite-vec` | **Low** | Auto-extension registration via FFI. Slightly fragile but well-documented. |

### What's Missing for a Real-Time Audio System

1. **No audio callback.** No `cpal::Stream`, no `jack::Client`, no `ASIOHandle`. The system exists entirely in the control plane.
2. **No lock-free data structures.** All cross-thread communication uses `tokio::sync::Mutex` and `tokio::sync::broadcast`. For a real-time audio thread, you need lock-free SPSC queues (`rtrb`, `ringbuf`).
3. **No dedicated real-time thread.** Everything runs on the tokio runtime with cooperative scheduling. A real-time audio thread needs `SCHED_FIFO` (Linux) or thread priority via `jack`.
4. **No MIDI clock sync.** No mention of MIDI clock (24 PPQN), song position pointer, or start/stop/continue messages.
5. **No latency measurement infrastructure.** No way to measure end-to-end latency from MIDI input → perception → directive → MIDI output. You can't improve what you can't measure.
6. **No jitter buffer.** Network MIDI or software-generated timing will have jitter. A jitter buffer (look-ahead scheduler) is essential for tight ensemble timing.
7. **No sample-accurate scheduling.** Note times are in beats (f32). For sample-accurate output, you need integer sample offsets and a sample-clock domain.

---

## Final Verdict

| Crate | Architecture | Production-Ready | Real-Time Ready | Key Risk |
|-------|-------------|-----------------|-----------------|----------|
| fleet-jepa-midi | Strong v1 | No (stub loops) | No (no I/O) | EMBEDDING_DIM mismatch |
| fleet-ensemble | Excellent design | No (stubs) | No (no I/O) | CNS bus priority inversion |
| fleet-gateway | Production-grade | **Yes** | N/A (control plane) | Mutex contention at scale |
| fleet-memory | Solid | **Yes** | N/A (offline) | DRY violations, O(n) ring buffers |

The fleet is a well-thought-out research prototype with one genuinely production-ready component (fleet-gateway) and one nearly-ready component (fleet-memory). The music crates need their v2 passes: unify the embedding dimension, add real MIDI I/O, separate the CNS bus by priority, and fix the spring-damper numerics.

The code is clean, well-commented, well-tested (every module has unit tests), and the architectural thinking is sound. The gap is between architecture and execution — the design docs describe a real-time system; the code describes a simulation of one.

That's OK. This is how you build complex systems: simulate first, wire later. Just don't ship it to a gig.

---

*End of diary. Review conducted with DeepSeek cross-verification on numerical stability claims and dependency risk assessment.*
