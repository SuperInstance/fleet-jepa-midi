#!/usr/bin/env python3
"""
elephant_sense_probe.py — prove the elephant (room-ness) is already latent in the
frozen audio-JEPA v2 encoder, WITHOUT retraining.

The captain's reframing: JEPA is a ROOM-TEMPERATURE SENSE. It is invisible from
inside a room; it is revealed by *contrast* — walking from one room into another
(sauna vs cold-plunge). This probe is the first decisive test of that claim.

WHAT IT DOES
 1. Loads the frozen v2 encoder (checkpoints/audio_jepa_v2.pt).
 2. Treats each radio-theater episode directory as a ROOM:
      - tap-trades/radio-theater/episode-{1,2,3,4}   (4 rooms, SAME cast — the
        killer control: same 6-7 speakers each night, different vibe)
      - radio-theater/compass-head-radio-hour/episode-* (up to 7 rooms)
 3. Extracts a 384-dim embedding per clip (mean-pool of window embeddings).
 4. Computes ELEPHANT METRICS (these replace the retired 0.849 ordering):
      (a) room discrimination accuracy (k-NN, with speaker-held-out control)
      (b) sauna/plunge separability (same-room vs cross-room cosine gap)
      (c) room "temperature" (spread) — mean pairwise distance *within* a room
      (d) speaker-confound ablation — does room signal survive after removing
          the speaker's own clips?

The decisive control: the tap-trades episodes share the SAME cast. If the
frozen encoder groups clips by *speaker* (voice identity) rather than by *room*,
k-NN room accuracy will collapse when we hold out a speaker. If it holds, the
elephant is real — the encoder feels something about the ROOM that is not just
"who is talking".

Usage:
    python3 elephant_sense_probe.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from audio_jepa.dataset import discover_clips, compute_mel
from audio_jepa.model import MelFrontend, ConvEncoder

# --------------------------------------------------------------------------- #
#  Room discovery
# --------------------------------------------------------------------------- #

def discover_rooms(roots: list[str], min_clips: int = 4) -> dict[str, list[Path]]:
    """Map room-name -> list of audio clips. A room = one episode directory.

    We ONLY take directories whose name matches `episode-*` — those are the true
    rooms (same cast, different night). Asset dirs (sfx/, voices/, segments/,
    songs/, mc/) are deliberately excluded: they are not rooms.
    """
    rooms: dict[str, list[Path]] = {}
    for root in roots:
        rp = Path(root)
        if not rp.is_dir():
            continue
        for d in sorted(rp.rglob("*")):
            if not d.is_dir():
                continue
            if not re.match(r"episode-?\d", d.name):
                continue
            clips = discover_clips(d)
            if len(clips) >= min_clips:
                rooms[str(d.relative_to(rp))] = clips
    return rooms


def speaker_key(filename: str) -> str:
    """A speaker/role key from the clip filename (tap-trades naming).

    lucineer-intro / wesley-here / carpenter-build / mason-patience / ... ->
    the leading token before the first '-'.
    """
    stem = Path(filename).stem.lower()
    # strip a leading episode/voice prefix if present
    stem = re.sub(r"^(episode-\d+[-_])", "", stem)
    # common suffixes that aren't speaker
    return stem.split("-")[0]


# --------------------------------------------------------------------------- #
#  Embedding extraction (reuse the v2 embed path)
# --------------------------------------------------------------------------- #

def load_encoder(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    encoder = ConvEncoder(
        n_mels=cfg.get("n_mels", 64),
        d_model=cfg.get("d_model", 256),
        n_layers=cfg.get("n_layers", 2),
        n_heads=cfg.get("n_heads", 4),
        latent_dim=cfg.get("latent_dim", 384),
    ).to(device)
    if "model" in ckpt:
        from audio_jepa.model import Predictor, AudioJEPA
        pred = Predictor(dim=encoder.latent_dim, hidden=cfg.get("predictor_hidden", 768))
        model = AudioJEPA(encoder, pred)
        model.load_state_dict(ckpt["model"])
    elif "encoder" in ckpt:
        encoder.load_state_dict(ckpt["encoder"])
    else:
        raise ValueError("checkpoint missing 'model' or 'encoder'")
    encoder.eval()
    return encoder, cfg


def embed_clip(path: Path, frontend, encoder, device, window=240, hop=120):
    import librosa
    y, sr = librosa.load(str(path), sr=16_000, mono=True)
    w = torch.from_numpy(y.astype(np.float32))
    mel = compute_mel(w, frontend, 16_000)  # CPU mel
    mel = mel.to(device)
    T = mel.shape[-1]
    if T < window:
        mel = F.pad(mel, (0, window - T))
        T = window
    m = mel.to(device)
    embs = []
    for s in range(0, T - window + 1, hop) or [0]:
        win = m[:, s:s + window].unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            z = encoder(win)
        embs.append(z[0])
    if not embs:
        with torch.no_grad():
            return F.normalize(encoder(m[:, :window].unsqueeze(0).unsqueeze(0))[0], dim=-1)
    return F.normalize(torch.stack(embs).mean(dim=0), dim=-1)


# --------------------------------------------------------------------------- #
#  Elephant metrics
# --------------------------------------------------------------------------- #

def cosine_matrix(Z: torch.Tensor) -> np.ndarray:
    Z = F.normalize(Z, dim=-1)
    return (Z @ Z.T).cpu().numpy()


def room_discrimination(Z, names, rooms, holdout_speaker=False) -> float:
    """k-NN (k=1) room accuracy. Optionally hold out the same speaker's clips.

    holdout_speaker=True removes, for each query clip, all clips sharing its
    speaker key from the candidate set (across ALL rooms). This is the decisive
    control: if room accuracy survives, the encoder feels ROOM not VOICE.
    """
    sim = cosine_matrix(Z)
    name_to_idx = {n: i for i, n in enumerate(names)}
    room_of = {n: r for r, nms in rooms.items() for n in nms}
    keys = {n: speaker_key(n) for n in names}
    correct = total = 0
    for i, n in enumerate(names):
        cands = [j for j in range(len(names)) if j != i]
        if holdout_speaker:
            cands = [j for j in cands if keys[names[j]] != keys[n]]
        if not cands:
            continue
        j = max(cands, key=lambda k: sim[i, k])
        correct += room_of[names[j]] == room_of[n]
        total += 1
    return correct / total if total else 0.0


def separability(Z, names, rooms) -> dict:
    """Same-room vs cross-room cosine: mean, gap, and a crude silhouette."""
    sim = cosine_matrix(Z)
    room_of = {n: r for r, nms in rooms.items() for n in nms}
    idx = {n: i for i, n in enumerate(names)}
    same, cross = [], []
    nms = list(names)
    for a in range(len(nms)):
        for b in range(a + 1, len(nms)):
            s = sim[a, b]
            if room_of[nms[a]] == room_of[nms[b]]:
                same.append(s)
            else:
                cross.append(s)
    same = np.array(same); cross = np.array(cross)
    return {
        "same_room_mean": float(same.mean()),
        "cross_room_mean": float(cross.mean()),
        "gap": float(same.mean() - cross.mean()),
        "n_same": int(len(same)),
        "n_cross": int(len(cross)),
    }


def room_temperature(Z, names, rooms) -> dict:
    """Room 'temperature' = mean pairwise distance (1-cos) within a room.
    A cold room = tight cluster (low spread); a warm/rowdy room = loose (high)."""
    sim = cosine_matrix(Z)
    room_of = {n: r for r, nms in rooms.items() for n in nms}
    idx = {n: i for i, n in enumerate(names)}
    out = {}
    for r, nms in rooms.items():
        ii = [idx[n] for n in nms if n in idx]
        if len(ii) < 2:
            continue
        ds = []
        for a in range(len(ii)):
            for b in range(a + 1, len(ii)):
                ds.append(1.0 - sim[ii[a], ii[b]])
        out[r] = float(np.mean(ds))
    return out


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/audio_jepa_v2.pt")
    ap.add_argument("--window", type=int, default=240)
    ap.add_argument("--hop", type=int, default=120)
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    print(f"[elephant] device={device}")

    encoder, cfg = load_encoder(args.ckpt, device)
    # keep the frontend on CPU: it produces the mel from a CPU waveform, and we
    # move the mel to the encoder's device afterwards (avoids a torchaudio STFT
    # device mismatch).
    frontend = MelFrontend(
        sample_rate=cfg.get("sample_rate", 16_000),
        n_fft=cfg.get("n_fft", 400),
        hop_length=cfg.get("hop_length", 160),
        n_mels=cfg.get("n_mels", 64),
    )

    roots = [
        "/home/eileen/projects/ai-writings/tap-trades/radio-theater",
        "/home/eileen/projects/ai-writings/radio-theater/compass-head-radio-hour",
    ]
    rooms = discover_rooms(roots)
    print(f"[elephant] discovered {len(rooms)} rooms:")
    for r, clips in sorted(rooms.items()):
        print(f"    {r:50s} {len(clips):3d} clips")

    # extract embeddings
    names, Zs, room_map = [], [], {}
    room_clips: dict[str, list[str]] = {}
    for r, clips in sorted(rooms.items()):
        room_clips[r] = []
        for p in clips:
            try:
                z = embed_clip(p, frontend, encoder, device, args.window, args.hop)
            except Exception as e:
                print(f"    !! skip {p.name}: {e}", file=sys.stderr)
                continue
            names.append(p.name)
            Zs.append(z)
            room_map[p.name] = r
            room_clips[r].append(p.name)
    Z = torch.stack(Zs)
    print(f"[elephant] embedded {len(names)} clips")

    # metrics
    print("\n" + "=" * 72)
    print("ELEPHANT METRICS (replace the retired 0.849 ordering)")
    print("=" * 72)

    acc_plain = room_discrimination(Z, names, room_clips, holdout_speaker=False)
    acc_held = room_discrimination(Z, names, room_clips, holdout_speaker=True)
    chance = 1.0 / max(1, len(rooms))
    print(f"\n  room discrimination (k-NN, no control):      {acc_plain:.3f}")
    print(f"  room discrimination (k-NN, SPEAKER-HELDOUT): {acc_held:.3f}")
    print(f"  chance (1 / n_rooms):                         {chance:.3f}")
    print(f"  -> if speaker-heldout stays >> chance, the elephant is REAL.")

    sep = separability(Z, names, room_clips)
    print(f"\n  sauna/plunge separability:")
    print(f"    same-room cosine mean:  {sep['same_room_mean']:.3f}")
    print(f"    cross-room cosine mean: {sep['cross_room_mean']:.3f}")
    print(f"    gap (separation):       {sep['gap']:.3f}")

    temp = room_temperature(Z, names, room_clips)
    print(f"\n  room temperature (mean within-room distance; cold=tight, warm=loose):")
    for r, t in sorted(temp.items(), key=lambda kv: -kv[1]):
        print(f"    {r:50s} {t:.3f}")

    out = {
        "n_rooms": len(rooms),
        "n_clips": len(names),
        "room_discrimination_plain": acc_plain,
        "room_discrimination_speaker_heldout": acc_held,
        "chance": chance,
        "separability": sep,
        "room_temperature": temp,
        "rooms": {r: len(c) for r, c in room_clips.items()},
    }
    Path("checkpoints/elephant_probe.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n[elephant] wrote checkpoints/elephant_probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
