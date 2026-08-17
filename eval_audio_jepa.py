#!/usr/bin/env python3
"""eval_audio_jepa.py — evaluate the learned audio ear against the hand-crafted one.

1. Loads the trained checkpoint.
2. Extracts a 384-dim embedding per clip (mean-pool of window embeddings).
3. Builds a greedy nearest-neighbour ordering using cosine similarity as the
   continuity signal (the same greedy procedure vibe_matcher.py uses, but with
   *learned* embeddings instead of hand-tuned acoustic deltas).
4. Compares against vibe_matcher.py's order.json via Kendall-tau and
   adjacent-pair overlap, plus a qualitative read of the two orderings.

Usage:
    python3 eval_audio_jepa.py --ckpt checkpoints/audio_jepa_v2.pt \
        --corpus /path/to/speeches --order /path/to/speeches/order.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from audio_jepa.dataset import discover_clips, compute_mel
from audio_jepa.model import MelFrontend, ConvEncoder, count_parameters


# --------------------------------------------------------------------------- #
#  Embedding extraction
# --------------------------------------------------------------------------- #

def load_model(ckpt_path: str, device):
    cfg = {}
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    encoder = ConvEncoder(
        n_mels=cfg.get("n_mels", 64),
        d_model=cfg.get("d_model", 256),
        n_layers=cfg.get("n_layers", 2),
        n_heads=cfg.get("n_heads", 4),
        latent_dim=cfg.get("latent_dim", 384),
    ).to(device)
    # the checkpoint saves the online model (encoder+predictor) and target_encoder
    if "model" in ckpt:
        # reconstruct full model then pull encoder weights
        from audio_jepa.model import Predictor, AudioJEPA
        pred = Predictor(dim=encoder.latent_dim, hidden=cfg.get("predictor_hidden", 768))
        model = AudioJEPA(encoder, pred)
        model.load_state_dict(ckpt["model"])
    elif "encoder" in ckpt:
        encoder.load_state_dict(ckpt["encoder"])
    else:
        raise ValueError("checkpoint missing 'model' or 'encoder' state")
    encoder.eval()
    return encoder, cfg


def embed_clip(mel: torch.Tensor, encoder, device, window: int, hop: int | None = None):
    """Mean-pool window embeddings of a [n_mels, T] mel into one [D] embedding."""
    hop = hop or window
    T = mel.shape[-1]
    if T < window:
        m = F.pad(mel, (0, window - T))
        T = window
    m = mel.to(device)
    embs = []
    starts = list(range(0, T - window + 1, hop))
    if not starts:
        starts = [0]
    for s in starts:
        win = m[:, s:s + window].unsqueeze(0).unsqueeze(0)  # [1,1,n_mels,w]
        with torch.no_grad():
            z = encoder(win)
        embs.append(z[0])
    embs = torch.stack(embs)  # [n_win, D]
    return F.normalize(embs.mean(dim=0), dim=-1)


# --------------------------------------------------------------------------- #
#  Ordering + comparison
# --------------------------------------------------------------------------- #

def greedy_order(names, sim_matrix, seed_idx=None):
    """Greedy nearest-neighbour ordering by similarity (mirrors vibe_matcher)."""
    n = len(names)
    if seed_idx is None:
        # most central: highest mean similarity to others
        seed_idx = int(np.argmax([sim_matrix[i].sum() for i in range(n)]))
    placed = [seed_idx]
    remaining = list(set(range(n)) - {seed_idx})
    while remaining:
        cur = placed[-1]
        nxt = max(remaining, key=lambda j: sim_matrix[cur, j])
        placed.append(nxt)
        remaining.remove(nxt)
    return [names[i] for i in placed]


def kendall_tau(a: list[str], b: list[str]) -> float:
    """Kendall tau-b between two rankings (permutations of the same items)."""
    assert set(a) == set(b), "rankings must be permutations of the same items"
    rank_b = {name: i for i, name in enumerate(b)}
    rank_a = {name: i for i, name in enumerate(a)}
    n = len(a)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            ai, aj = rank_a[a[i]], rank_a[a[j]]
            bi, bj = rank_b[a[i]], rank_b[a[j]]
            if (ai - aj) * (bi - bj) > 0:
                concordant += 1
            elif (ai - aj) * (bi - bj) < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 0.0


def adjacent_pairs(order: list[str]) -> set:
    return {frozenset((order[i], order[i + 1])) for i in range(len(order) - 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/audio_jepa_v2.pt")
    ap.add_argument("--corpus", default="/home/eileen/projects/ai-writings/speeches")
    ap.add_argument("--order", default="/home/eileen/projects/ai-writings/speeches/order.json")
    ap.add_argument("--out", default="checkpoints/eval_output.json")
    ap.add_argument("--window", type=int, default=240)
    ap.add_argument("--hop", type=int, default=120)
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    print(f"[eval] device={device}")

    encoder, cfg = load_model(args.ckpt, device)
    print(f"[eval] encoder params {count_parameters(encoder)/1e6:.2f}M "
          f"(latent {cfg.get('latent_dim', 384)})")

    clips = discover_clips(args.corpus)
    print(f"[eval] {len(clips)} clips")

    frontend = MelFrontend()
    import librosa
    import numpy as _np
    names = []
    embs = []
    tail_embs = []
    head_embs = []
    for p in clips:
        y, sr = librosa.load(str(p), sr=16_000, mono=True)
        w = torch.from_numpy(y.astype(_np.float32))
        mel = compute_mel(w, frontend, 16_000)
        emb = embed_clip(mel, encoder, device, args.window, args.hop)
        # directional: tail = last window, head = first window
        tail = embed_clip(mel[:, -args.window:], encoder, device, args.window, args.window)
        head = embed_clip(mel[:, : args.window], encoder, device, args.window, args.window)
        names.append(p.name)
        embs.append(emb)
        tail_embs.append(tail)
        head_embs.append(head)
        print(f"    {p.name:50s}  |emb|={emb.norm().item():.3f}")

    embs = torch.stack(embs).cpu()
    tail_embs = torch.stack(tail_embs).cpu()
    head_embs = torch.stack(head_embs).cpu()

    # pairwise cosine similarity matrices
    sim = embs @ embs.T
    sim_dir = tail_embs @ head_embs.T  # sim(A->B) = cos(tail_A, head_B)

    learned_order = greedy_order(names, sim.numpy())
    learned_order_dir = greedy_order(names, sim_dir.numpy())

    # load hand-crafted order
    order = json.loads(Path(args.order).read_text())
    hand_order = order["sequence"]

    # comparison (on the set of clips the learned eval actually saw)
    # vibe_matcher ran on 16 clips; ensure name sets match
    common = [n for n in names if n in set(hand_order)]
    if len(common) != len(names):
        print(f"[eval] WARNING: clip name mismatch: {len(names)} learned vs "
              f"{len(hand_order)} hand-crafted; comparing over {len(common)} common")

    tau_global = kendall_tau(names, hand_order) if set(names) == set(hand_order) else float("nan")
    tau_dir = kendall_tau(learned_order_dir, hand_order) if set(names) == set(hand_order) else float("nan")

    # adjacent-pair overlap
    adj_hand = adjacent_pairs(hand_order)
    adj_learned = adjacent_pairs(learned_order)
    adj_learned_dir = adjacent_pairs(learned_order_dir)
    overlap = len(adj_hand & adj_learned) / max(1, len(adj_hand))
    overlap_dir = len(adj_hand & adj_learned_dir) / max(1, len(adj_hand))

    # similarity: mean cosine similarity of the hand-crafted adjacent pairs
    rank = {n: i for i, n in enumerate(names)}
    hand_adj_sim = []
    for a, b in zip(hand_order[:-1], hand_order[1:]):
        if a in rank and b in rank:
            hand_adj_sim.append(float(sim[rank[a], rank[b]]))
    mean_hand_adj_sim = float(np.mean(hand_adj_sim)) if hand_adj_sim else float("nan")

    # variance stats of the embedding space (collapse check)
    std_z = embs.std(dim=0).mean().item()
    mean_pairwise_sim = float(sim[~np.eye(len(names), dtype=bool)].mean())

    result = {
        "checkpoint": args.ckpt,
        "clip_count": len(names),
        "latent_dim": cfg.get("latent_dim", 384),
        "embedding_std_per_dim": round(std_z, 4),
        "mean_pairwise_cosine_similarity": round(mean_pairwise_sim, 4),
        "learned_order_global": learned_order,
        "learned_order_directional": learned_order_dir,
        "hand_crafted_order": hand_order,
        "kendall_tau_global": round(tau_global, 4) if not math.isnan(tau_global) else None,
        "kendall_tau_directional": round(tau_dir, 4) if not math.isnan(tau_dir) else None,
        "adjacent_pair_overlap_global": round(overlap, 4),
        "adjacent_pair_overlap_directional": round(overlap_dir, 4),
        "mean_similarity_of_hand_adjacent_pairs": round(mean_hand_adj_sim, 4),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\n[eval] wrote -> {out}")

    print("\n[eval] ===== RESULTS =====")
    print(f"  embedding std/coord        : {std_z:.4f} (collapse if ~0)")
    print(f"  mean pairwise cosine sim   : {mean_pairwise_sim:.4f}")
    print(f"  Kendall-tau (global)       : {tau_global:+.3f}")
    print(f"  Kendall-tau (directional)  : {tau_dir:+.3f}")
    print(f"  adjacent-pair overlap      : {overlap:.3f} (global) / {overlap_dir:.3f} (dir)")
    print(f"  mean sim of hand adj pairs : {mean_hand_adj_sim:.3f}")
    print(f"\n  learned order (global):")
    print("    " + " -> ".join(learned_order[:8]) + " ...")
    print(f"  hand-crafted order:")
    print("    " + " -> ".join(hand_order[:8]) + " ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
