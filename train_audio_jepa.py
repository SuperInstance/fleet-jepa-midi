#!/usr/bin/env python3
"""train_audio_jepa.py — self-supervised training of the learned audio ear (v2).

Trains a JEPA over log-mel spectrograms of the 16-clip speeches corpus:
    context (past, 50% of frames masked) -> online encoder -> predictor
    target  (future)                      -> EMA encoder   (stop-gradient)

Loss = cosine distance(prediction, stopgrad(target)) + VICReg(variance + covariance).

Usage:
    python3 train_audio_jepa.py --corpus /path/to/speeches --epochs 100
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from audio_jepa.dataset import SpeechClipsDataset
from audio_jepa.model import MelFrontend, build_model, count_parameters


# --------------------------------------------------------------------------- #
#  VICReg
# --------------------------------------------------------------------------- #

def _off_diag_cov(h: torch.Tensor) -> torch.Tensor:
    """Mean squared off-diagonal covariance of [B, D]."""
    h = h - h.mean(dim=0)
    B, D = h.shape
    cov = (h.T @ h) / (B - 1)
    mask = ~torch.eye(D, dtype=torch.bool, device=cov.device)
    return (cov[mask] ** 2).mean()


def vicreg_loss(h1: torch.Tensor, h2: torch.Tensor,
                gamma: float = 1.0, lambda_var: float = 1.0,
                lambda_cov: float = 1.0, eps: float = 1e-4):
    """VICReg variance + covariance over two branches' raw projections."""
    std1 = torch.sqrt(h1.var(dim=0) + eps)
    std2 = torch.sqrt(h2.var(dim=0) + eps)
    var = (F.relu(gamma - std1).mean() + F.relu(gamma - std2).mean())
    cov = _off_diag_cov(h1) + _off_diag_cov(h2)
    return lambda_var * var, lambda_cov * cov


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def make_mask(B: int, T: int, ratio: float, device, seed_fn) -> torch.Tensor:
    """Boolean-ish mask [B, T]: 1=keep, 0=masked. Masks ~ratio of frames."""
    keep = torch.rand(B, T, device=device, generator=seed_fn) > ratio
    return keep.float()


def ema_tau(step: int, total: int, base: float = 0.99, max_tau: float = 0.999) -> float:
    """Momentum schedule: ramp base -> max_tau (cosine). Never reaches 1.0 so the
    EMA target always slowly follows the online encoder (anti-collapse)."""
    p = min(step / max(1, total), 1.0)
    tau = 1.0 - (1.0 - base) * (math.cos(math.pi * p) + 1.0) / 2.0
    return min(tau, max_tau)


def update_ema(target: torch.nn.Module, online: torch.nn.Module, tau: float):
    with torch.no_grad():
        for tp, op in zip(target.parameters(), online.parameters()):
            tp.data.mul_(tau).add_(op.data, alpha=1.0 - tau)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/home/eileen/projects/ai-writings/speeches")
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--steps-per-epoch", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mask-ratio", type=float, default=0.5)
    ap.add_argument("--lambda-var", type=float, default=1.0)
    ap.add_argument("--lambda-cov", type=float, default=0.2)
    ap.add_argument("--lambda-inv", type=float, default=1.0)
    ap.add_argument("--ema-base", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-cuda", action="store_true")
    ap.add_argument("--amp", action="store_true", help="use fp16 autocast")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    print(f"[train] device={device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # dataset (precomputes log-mels; augmentations applied in mel domain)
    # frontend stays on CPU: it is only used once to build the mel cache
    frontend = MelFrontend()
    ds = SpeechClipsDataset(args.corpus, frontend, sample_rate=16_000, seed=args.seed)
    print(f"[train] corpus: {len(ds.clips)} clips, "
          f"total {sum(ds.durations)/60:.1f} min audio")

    # model
    model, target_encoder = build_model(device=device)
    print(f"[train] online params: {count_parameters(model)/1e6:.2f}M "
          f"(encoder {count_parameters(model.encoder)/1e6:.2f}M + "
          f"predictor {count_parameters(model.predictor)/1e6:.2f}M)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    total_steps = args.epochs * args.steps_per_epoch
    warmup = args.steps_per_epoch * 5

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * p))

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    log_path = out_dir / "train_log.csv"
    f = open(log_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["step", "epoch", "loss", "inv", "var", "cov",
                     "lr", "ema_tau", "std_z", "std_z_t", "batch_sim", "sec_per_step"])

    step = 0
    t0 = time.time()
    print(f"[train] {args.epochs} epochs x {args.steps_per_epoch} steps "
          f"= {total_steps} steps")

    for epoch in range(args.epochs):
        for _ in range(args.steps_per_epoch):
            step += 1
            # sample a batch of mel windows
            ctx_wins, tgt_wins = [], []
            for _ in range(args.batch_size):
                c, t = ds.sample_window()
                ctx_wins.append(c)
                tgt_wins.append(t)
            ctx_mel = torch.stack(ctx_wins).to(device)   # [B, n_mels, ctx]
            tgt_mel = torch.stack(tgt_wins).to(device)   # [B, n_mels, tgt]
            ctx_mel = ctx_mel.unsqueeze(1)               # [B, 1, n_mels, ctx]
            tgt_mel = tgt_mel.unsqueeze(1)
            B, _, Fm, Tm = ctx_mel.shape

            # random frame masking on the context input
            g = torch.Generator(device="cpu")
            g.manual_seed(step * 7919 + args.seed)
            mask = make_mask(B, Tm, args.mask_ratio, "cpu", g).to(device)

            with torch.cuda.amp.autocast(enabled=args.amp):
                pred, z_c, h_c = model(ctx_mel, mask=mask, return_raw=True)
                with torch.no_grad():
                    z_t, h_t = target_encoder(tgt_mel, return_raw=True)
                    z_t = z_t.detach()
                    h_t = h_t.detach()

                # invariance: cosine distance between prediction and target
                inv = (1.0 - F.cosine_similarity(pred, z_t, dim=-1)).mean()
                # VICReg variance+covariance on the batch-normalized projector
                # output h (BatchNorm in the projector is the primary collapse
                # guard; the variance term then only prevents BN-scale collapse).
                var, cov = vicreg_loss(h_c, h_t, gamma=1.0,
                                       lambda_var=args.lambda_var,
                                       lambda_cov=args.lambda_cov)
                loss = args.lambda_inv * inv + var + cov

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            # update EMA target
            tau = ema_tau(step, total_steps, base=args.ema_base)
            update_ema(target_encoder, model.encoder, tau)

            # per-step lr (set after warmup)
            lr = lr_at(step)
            for pg in opt.param_groups:
                pg["lr"] = lr

            if step % 25 == 0 or step == 1:
                dt = (time.time() - t0) / max(1, step)
                std_z = z_c.std(dim=0).mean().item()
                std_z_t = z_t.std(dim=0).mean().item()
                # mean off-diagonal cosine similarity within the batch (collapse metric)
                sm = z_c @ z_c.T
                off = ~torch.eye(B, dtype=torch.bool, device=sm.device)
                batch_sim = sm[off].mean().item()
                writer.writerow([step, epoch, loss.item(), inv.item(),
                                 var.item(), cov.item(), lr, tau,
                                 std_z, std_z_t, batch_sim, round(dt, 4)])
                f.flush()
                print(f"  step {step:5d}/{total_steps}  loss {loss.item():.4f} "
                      f"(inv {inv.item():.4f} var {var.item():.4f} cov {cov.item():.4f}) "
                      f"lr {lr:.2e} tau {tau:.3f} std_z {std_z:.4f} sim {batch_sim:.3f}")

        # end of epoch summary
        print(f"[train] epoch {epoch+1}/{args.epochs} done "
              f"({time.time()-t0:.1f}s elapsed)")

    f.close()
    train_time = time.time() - t0

    # save checkpoint
    ckpt = out_dir / "audio_jepa_v2.pt"
    torch.save({
        "model": model.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "config": {
            "n_mels": 64, "d_model": 256, "n_layers": 2, "n_heads": 4,
            "latent_dim": 384, "predictor_hidden": 768,
            "sample_rate": 16000, "context_frames": ds.context_frames,
            "target_frames": ds.target_frames,
            "mask_ratio": args.mask_ratio,
            "lambda_var": args.lambda_var, "lambda_cov": args.lambda_cov,
            "lambda_inv": args.lambda_inv, "ema_base": args.ema_base,
        },
        "epochs": args.epochs, "steps": total_steps,
        "train_time_sec": round(train_time, 1),
    }, ckpt)
    print(f"[train] saved checkpoint -> {ckpt}")
    print(f"[train] done in {train_time:.1f}s")

    # loss curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps, loss, inv, var, cov = [], [], [], [], []
        with open(log_path) as fh:
            rd = csv.DictReader(fh)
            for row in rd:
                steps.append(int(row["step"])); loss.append(float(row["loss"]))
                inv.append(float(row["inv"])); var.append(float(row["var"]))
                cov.append(float(row["cov"]))
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(steps, loss, label="total", lw=1)
        ax[0].plot(steps, inv, label="invariance", lw=1, alpha=0.6)
        ax[0].set_xlabel("step"); ax[0].set_ylabel("loss"); ax[0].legend()
        ax[0].set_title("JEPA training loss")
        ax[1].plot(steps, var, label="variance", lw=1)
        ax[1].plot(steps, cov, label="covariance", lw=1)
        ax[1].set_xlabel("step"); ax[1].set_ylabel("loss"); ax[1].legend()
        ax[1].set_title("VICReg terms")
        fig.tight_layout()
        png = out_dir / "loss_curve.png"
        fig.savefig(png, dpi=110)
        print(f"[train] wrote loss curve -> {png}")
    except Exception as e:  # pragma: no cover
        print(f"[train] (loss plot skipped: {e})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
