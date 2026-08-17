#!/usr/bin/env python3
"""
vibe_matcher.py — the fleet's PERCEPTION ENGINE (v1, acoustic stand-in for JEPA).

The Tap radio theater renders many separate audio clips (different TTS voices,
different moments). Separately they are a playlist; together they should be ONE
show. This tool is the "ear" that makes the station coherent:

  1. LISTENS  — analyses every rendered clip in a directory (duration, loudness
     profile, energy curve, spectral brightness/tilt, flatness, zero-crossing
     rate, tempo, MFCC timbre) and writes `clip-manifest.json`.
  2. FEELS    — computes a directional "vibe-continuity" score between every
     ordered pair of clips: how smoothly A's *tail* flows into B's *head*
     (loudness delta, brightness delta, pace delta, timbre cosine-similarity,
     and — when a local embedding model + transcript exist — semantic similarity).
  3. ORDERS   — greedily chains the clips (nearest-neighbour by continuity) into
     one optimal order, and suggests a cross-fade duration per transition.
     Writes `order.json`.

This v1 is a hand-crafted acoustic feature vector — the same pattern the
fleet-jepa-midi repo uses for MIDI (16 hand-crafted features -> projected
embedding), applied to AUDIO. The roadmap to a true learned audio-JEPA is in
research/vibe-matcher-2026-08-16.md.

Dependencies: librosa, numpy, soundfile (all present on the host).
Optional semantic ear: a local Ollama `nomic-embed-text` model + a matching
`.md` transcript in the corpus dir.

Usage:
    python3 vibe_matcher.py --dir /path/to/speeches
    python3 vibe_matcher.py --dir /path/to/speeches --no-semantic
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

SR = 22050                 # analysis sample rate (mono)
ENERGY_BINS = 20           # energy curve resolution
SILENCE_RMS = 0.008        # RMS below this counts as "no energy" (for cross-fade)
MIN_FILE_BYTES = 50_000    # skip placeholder/tiny files


# --------------------------------------------------------------------------- #
#  Audio feature extraction (the acoustic "ear")
# --------------------------------------------------------------------------- #

def _db(rms: float) -> float:
    return float(20.0 * np.log10(rms + 1e-10))


def extract_features(path: Path) -> Optional[dict]:
    """Analyse one audio file into a rich acoustic feature dict."""
    try:
        import librosa  # local import keeps error surface clear
    except Exception as e:  # pragma: no cover
        print(f"  !! librosa unavailable: {e}", file=sys.stderr)
        raise

    try:
        y, sr = librosa.load(str(path), sr=SR, mono=True)
    except Exception as e:
        print(f"  !! failed to decode {path.name}: {e}", file=sys.stderr)
        return None

    duration = float(len(y) / sr)
    if duration < 0.4:
        return None

    # --- loudness / energy ---
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_mean = float(rms.mean())
    rms_std = float(rms.std())
    rms_max = float(rms.max())

    # energy curve: mean RMS in ENERGY_BINS equal time slices
    slices = np.array_split(rms, ENERGY_BINS)
    energy_curve = [float(s.mean()) for s in slices]

    # head / tail energy (first & last ~2s) — the cross-fade-relevant signal
    head = rms[: int(2.0 * sr / 512)]
    tail = rms[-int(2.0 * sr / 512):]
    head_energy = float(head.mean()) if len(head) else 0.0
    tail_energy = float(tail.mean()) if len(tail) else 0.0

    # --- spectral shape ---
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flat = librosa.feature.spectral_flatness(y=y)[0]

    # spectral tilt: slope of log-magnitude vs log-frequency (dB per decade-ish)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mag = S.mean(axis=1)
    band = (freqs >= 60.0) & (freqs <= 8000.0)
    if band.sum() > 8:
        tilt = float(np.polyfit(np.log10(freqs[band]),
                                np.log10(mag[band] + 1e-10), 1)[0])
    else:
        tilt = 0.0

    zcr = librosa.feature.zero_crossing_rate(y)[0]

    # --- tempo ---
    try:
        bt = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.asarray(bt[0] if isinstance(bt, tuple) else bt).ravel()[0])
    except Exception:
        tempo = None

    # --- timbre: MFCC ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # --- naive speech-vs-music heuristic ---
    # speech: high flatness, high-ish zcr, energy concentrated in 80-4000Hz;
    # music: lower flatness, sustained harmonic content. filename hint augments it.
    name = path.stem.lower()
    if re.search(r"(song|underscore|music|instrumental)", name):
        kind = "music"
    elif float(flat.mean()) > 0.02 and float(cent.mean()) < 3200.0:
        kind = "speech"
    else:
        kind = "speech"  # TTS corpus is speech-first; keep the fallback simple

    return {
        "filename": path.name,
        "duration_sec": round(duration, 3),
        "kind": kind,
        "loudness": {
            "mean_rms": round(rms_mean, 5),
            "std_rms": round(rms_std, 5),
            "max_rms": round(rms_max, 5),
            "mean_dbfs": round(_db(rms_mean), 2),
            "max_dbfs": round(_db(rms_max), 2),
        },
        "energy_curve": [round(v, 5) for v in energy_curve],
        "head_energy": round(head_energy, 5),
        "tail_energy": round(tail_energy, 5),
        "spectral": {
            "centroid_mean": round(float(cent.mean()), 2),
            "centroid_std": round(float(cent.std()), 2),
            "rolloff_mean": round(float(roll.mean()), 2),
            "flatness_mean": round(float(flat.mean()), 5),
            "tilt": round(tilt, 4),
        },
        "zcr_mean": round(float(zcr.mean()), 5),
        "tempo_bpm": round(tempo, 2) if tempo else None,
        "mfcc_mean": [round(float(v), 4) for v in mfcc.mean(axis=1)],
        "mfcc_std": [round(float(v), 4) for v in mfcc.std(axis=1)],
    }


# --------------------------------------------------------------------------- #
#  Semantic ear (optional): local nomic-embed-text over the transcript
# --------------------------------------------------------------------------- #

AUDIO_SUFFIXES = [
    "-underscore-local", "-vhf-gateway", "-qwen-tts", "-underscore",
    "-vhf-tts", "-v2-tts", "-local", "-gateway", "-tts", "-v2",
]


def base_key(stem: str) -> str:
    """Strip known audio suffixes from a clip stem to get its content key."""
    key = stem
    for suf in AUDIO_SUFFIXES:
        if key.endswith(suf):
            key = key[: -len(suf)]
    return key


def find_transcript(clip_stem: str, md_stems: list[str]) -> Optional[str]:
    """Best-effort match a clip to its source transcript .md stem."""
    clip_stem = Path(clip_stem).stem  # drop any extension
    key = base_key(clip_stem)
    # song-N-* maps to song-N-lyrics-*
    m = re.match(r"(song-\d+)-", key)
    song_key = m.group(1) if m else None

    best, best_len = None, 0
    for md in md_stems:
        lmd = md.lower()
        if lmd == key:
            return md
        if key and (key in lmd or lmd in key) and len(md) > best_len:
            best, best_len = md, len(md)
        if song_key and song_key in lmd and len(md) > best_len:
            best, best_len = md, len(md)
    return best


def ollama_embed(text: str, model: str = "nomic-embed-text") -> Optional[list[float]]:
    """Get a local embedding vector from a running Ollama instance."""
    payload = json.dumps({"model": model, "input": text}).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/embed",
        data=payload, headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode())
        emb = data.get("embeddings", [None])[0]
        return list(emb) if emb is not None else None
    except Exception as e:
        print(f"  !! ollama embed failed ({e})", file=sys.stderr)
        return None


# --------------------------------------------------------------------------- #
#  Vibe-continuity scoring
# --------------------------------------------------------------------------- #

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _norm01(x: float, lo: float, hi: float) -> float:
    """Map x into [0,1] against a [lo, hi] range, clamped."""
    if hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def timbre_vector(c: dict) -> np.ndarray:
    return np.asarray(
        c["mfcc_mean"] + c["mfcc_std"]
        + [c["spectral"]["tilt"], c["spectral"]["flatness_mean"],
           c["zcr_mean"], c["spectral"]["centroid_mean"] / 8000.0,
           c["spectral"]["centroid_std"] / 8000.0,
           c["spectral"]["rolloff_mean"] / 8000.0],
        dtype=float,
    )


def continuity(a: dict, b: dict, stds: dict, sem: dict) -> tuple[float, str]:
    """
    Directional vibe-continuity A -> B (A's tail into B's head).
    Returns (score in [0,1], human-readable reason).
    """
    comps = {}

    # 1) boundary energy gap (tail of A vs head of B) — the cross-fade signal
    energy_hi = stds["head_tail_max"] or 0.2
    gap = abs(b["head_energy"] - a["tail_energy"]) / energy_hi
    comps["boundary_energy"] = 1.0 - _norm01(gap, 0.0, 1.0)

    # 2) loudness delta (overall level continuity)
    loud_hi = stds["rms_max"] - stds["rms_min"] or 0.1
    ld = abs(b["loudness"]["mean_rms"] - a["loudness"]["mean_rms"]) / loud_hi
    comps["loudness"] = 1.0 - _norm01(ld, 0.0, 1.0)

    # 3) brightness delta
    b_hi = stds["centroid_max"] - stds["centroid_min"] or 4000.0
    bd = abs(b["spectral"]["centroid_mean"] - a["spectral"]["centroid_mean"]) / b_hi
    comps["brightness"] = 1.0 - _norm01(bd, 0.0, 1.0)

    # 4) pace delta (tempo)
    if a["tempo_bpm"] and b["tempo_bpm"]:
        pd = abs(b["tempo_bpm"] - a["tempo_bpm"]) / 120.0
        comps["pace"] = 1.0 - _norm01(pd, 0.0, 1.0)
    else:
        comps["pace"] = 0.5

    # 5) timbre cosine similarity (MFCC + spectral shape)
    comps["timbre"] = cosine(timbre_vector(a), timbre_vector(b))

    # 6) semantic similarity (only when both clips have a text embedding)
    ea, eb = sem.get(a["filename"]), sem.get(b["filename"])
    if ea is not None and eb is not None:
        comps["semantic"] = cosine(np.asarray(ea), np.asarray(eb))

    weights = {
        "boundary_energy": 0.30,
        "loudness": 0.15,
        "brightness": 0.10,
        "pace": 0.10,
        "timbre": 0.25,
        "semantic": 0.10,
    }
    # renormalise weights over the components actually present
    present = {k: w for k, w in weights.items() if k in comps}
    tot = sum(present.values())
    score = sum(comps[k] * w / tot for k, w in present.items())

    # reason = the weakest (most penalised) component — what the ear noticed
    weakest = min(present, key=lambda k: comps[k])
    return float(np.clip(score, 0.0, 1.0)), weakest


def crossfade_ms(a: dict, b: dict, cont: float) -> int:
    """Suggested cross-fade length (ms) for an A -> B transition."""
    # silence at either boundary -> a clean cut, no fade
    if a["tail_energy"] < SILENCE_RMS or b["head_energy"] < SILENCE_RMS:
        return 0
    # smoother transitions get a shorter fade; jarring ones get a longer blend.
    # music boundaries are allowed to breathe a little longer.
    bonus = 400 if (a["kind"] == "music" or b["kind"] == "music") else 0
    ms = int(round(300 + 2000 * (1.0 - cont) + bonus))
    return int(np.clip(ms, 0, 2500) // 25 * 25)  # snap to 25ms


# --------------------------------------------------------------------------- #
#  Ordering (greedy nearest-neighbour)
# --------------------------------------------------------------------------- #

def optimal_order(clips: list[dict], stds: dict, sem: dict,
                  seed: Optional[str]) -> tuple[list[str], list[dict], float]:
    """Chain clips by greedy nearest-neighbour on vibe-continuity."""
    names = [c["filename"] for c in clips]

    # directional continuity matrix
    cont = {}
    for a in clips:
        for b in clips:
            if a["filename"] == b["filename"]:
                continue
            cont[(a["filename"], b["filename"])], _ = continuity(a, b, stds, sem)

    # seed: most central clip (highest mean outgoing continuity) unless given
    if seed and seed in names:
        first = seed
    else:
        first = max(names, key=lambda n: float(
            np.mean([cont[(n, m)] for m in names if m != n])))

    placed = [first]
    remaining = [n for n in names if n != first]
    transitions: list[dict] = []

    while remaining:
        cur = placed[-1]
        nxt = max(remaining, key=lambda n: cont[(cur, n)])
        score, reason = continuity(
            next(c for c in clips if c["filename"] == cur),
            next(c for c in clips if c["filename"] == nxt),
            stds, sem,
        )
        fade = crossfade_ms(
            next(c for c in clips if c["filename"] == cur),
            next(c for c in clips if c["filename"] == nxt),
            score,
        )
        transitions.append({
            "from": cur, "to": nxt,
            "continuity": round(score, 4),
            "crossfade_ms": fade,
            "weakest_signal": reason,
        })
        placed.append(nxt)
        remaining.remove(nxt)

    total = float(np.mean([t["continuity"] for t in transitions])) if transitions else 0.0
    return placed, transitions, total


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="Perception engine: vibe-match rendered clips into one show.")
    ap.add_argument("--dir", default="/home/eileen/projects/ai-writings/speeches",
                    help="directory of rendered audio clips")
    ap.add_argument("--out-dir", default=None, help="where to write manifests (default: --dir)")
    ap.add_argument("--seed", default=None, help="filename to open the show with (default: most central)")
    ap.add_argument("--no-semantic", action="store_true",
                    help="skip the local nomic-embed-text semantic ear")
    args = ap.parse_args()

    corpus = Path(args.dir)
    if not corpus.is_dir():
        print(f"!! not a directory: {corpus}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else corpus

    # discover clips
    audio_files = sorted(
        p for p in corpus.iterdir()
        if p.suffix.lower() in (".mp3", ".wav", ".m4a", ".flac", ".ogg")
        and p.stat().st_size >= MIN_FILE_BYTES
    )
    if not audio_files:
        print(f"!! no audio files (>= {MIN_FILE_BYTES}B) in {corpus}", file=sys.stderr)
        return 1

    print(f"[perception] listening to {len(audio_files)} clips in {corpus}")
    clips = []
    for i, p in enumerate(audio_files, 1):
        print(f"  [{i:>2}/{len(audio_files)}] {p.name}")
        feats = extract_features(p)
        if feats:
            clips.append(feats)

    if not clips:
        print("!! no clips decoded", file=sys.stderr)
        return 1
    print(f"[perception] decoded {len(clips)} clips")

    # --- semantic ear ---
    sem: dict[str, list[float]] = {}
    if not args.no_semantic:
        md_stems = [m.stem for m in corpus.glob("*.md")]
        print("[semantic] checking local nomic-embed-text over transcripts...")
        got = 0
        for c in clips:
            if c["kind"] != "speech":
                continue
            md = find_transcript(c["filename"], md_stems)
            if not md:
                continue
            txt = (corpus / f"{md}.md").read_text(encoding="utf-8", errors="ignore")
            emb = ollama_embed(txt[:4000])
            if emb is not None:
                sem[c["filename"]] = emb
                c["text_source"] = f"{md}.md"
                c["has_text_embedding"] = True
                got += 1
            else:
                c["text_source"] = f"{md}.md"
                c["has_text_embedding"] = False
        print(f"[semantic] text embeddings for {got} clip(s)")
    else:
        for c in clips:
            c["has_text_embedding"] = False

    # corpus-wide ranges for normalisation
    stds = {
        "rms_min": min(c["loudness"]["mean_rms"] for c in clips),
        "rms_max": max(c["loudness"]["mean_rms"] for c in clips),
        "centroid_min": min(c["spectral"]["centroid_mean"] for c in clips),
        "centroid_max": max(c["spectral"]["centroid_mean"] for c in clips),
        "head_tail_max": max(max(c["head_energy"], c["tail_energy"]) for c in clips) or 0.2,
    }

    # --- write clip manifest ---
    manifest = {
        "generated_at": _now(),
        "corpus_dir": str(corpus),
        "analysis_sr": SR,
        "energy_bins": ENERGY_BINS,
        "clip_count": len(clips),
        "clips": clips,
    }
    mpath = out_dir / "clip-manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[perception] wrote {mpath}")

    # --- order ---
    seq, transitions, total = optimal_order(clips, stds, sem, args.seed)
    order = {
        "generated_at": _now(),
        "corpus_dir": str(corpus),
        "method": "greedy-nearest-neighbour by vibe-continuity (directional A->B)",
        "seed": seq[0],
        "sequence": seq,
        "transitions": transitions,
        "total_continuity": round(total, 4),
        "avg_crossfade_ms": int(round(np.mean([t["crossfade_ms"] for t in transitions]))) if transitions else 0,
    }
    opath = out_dir / "order.json"
    opath.write_text(json.dumps(order, indent=2) + "\n")
    print(f"[perception] wrote {opath}")

    # --- summary ---
    print("\n[show] optimal order (first 8):")
    for n in seq[:8]:
        c = next(x for x in clips if x["filename"] == n)
        print(f"    {c['duration_sec']:>7.1f}s  {c['kind']:<6}  {n}")
    print(f"\n[show] first 5 transitions:")
    for t in transitions[:5]:
        print(f"    {t['from']}  ->  {t['to']}")
        print(f"        continuity {t['continuity']:.3f}  crossfade {t['crossfade_ms']}ms  (weakest: {t['weakest_signal']})")
    print(f"\n[show] total continuity {total:.3f} over {len(transitions)} transitions, "
          f"avg crossfade {order['avg_crossfade_ms']}ms")
    return 0


def _now() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
