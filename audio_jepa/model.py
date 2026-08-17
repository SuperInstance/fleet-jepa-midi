"""audio_jepa/model.py — mel frontend + conv/Conformer encoder + JEPA predictor.

Architecture (mirrors README target, scaled down for a 16-clip corpus):

    raw waveform (mono, 16 kHz)
        -> MelFrontend (log-mel spectrogram, 64 mels, 100 fps)
        -> ConvEncoder (4 conv blocks w/ BN+GELU, freq pooling)
        -> TransformerEncoder (2 layers, 4 heads, d=256)   [Conformer-lite]
        -> mean-pool over time -> Linear(256 -> 384) -> L2-normalize
        -> Predictor (BYOL-style MLP 384 -> 768 -> 384, L2-normalize)

The online encoder `f_theta` and the EMA target encoder `f_theta'` share the
same architecture. Only the online encoder receives gradients; the target is
updated by exponential moving average (momentum schedule). The predictor maps
the *context* embedding onto the *target* embedding; the loss is cosine
distance between the normalized prediction and the stop-gradient target.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# --------------------------------------------------------------------------- #
#  Frontend
# --------------------------------------------------------------------------- #


class MelFrontend(nn.Module):
    """Log-mel spectrogram frontend. Input: [B, T_samples] or [B, 1, T_samples]."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 64,
        f_min: float = 50.0,
        f_max: float = 8000.0,
        top_db: float = 80.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
            normalized=False,
            center=True,
            pad_mode="reflect",
        )
        self.top_db = top_db

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T] or [B, 1, T] -> [B, 1, n_mels, frames] (log-mel, dB)."""
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, T]
        spec = self.mel(x)  # [B, 1, n_mels, frames]
        # amplitude -> dB, clamp to top_db dynamic range, normalize to ~[-1, 1]
        spec = torch.clamp(spec, min=1e-10)
        spec = 10.0 * torch.log10(spec)  # dB power
        ref = spec.max() if False else 0.0  # keep absolute dB; normalize below
        spec = torch.clamp(spec, min=spec.max() - self.top_db)  # clip bottom
        # normalize to zero-mean-ish unit-ish: center by global mean, scale
        return spec  # caller normalizes via LayerNorm-free conv BN


# --------------------------------------------------------------------------- #
#  Encoder
# --------------------------------------------------------------------------- #


def _sinusoidal_pos(dim: int, length: int, device=None, dtype=None) -> torch.Tensor:
    """Sinusoidal positional embedding [length, dim]."""
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(dim, device=device, dtype=torch.float32).unsqueeze(0)
    angle = pos / (10_000 ** (2 * (i // 2) / dim))
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(angle[:, 0::2])
    pe[:, 1::2] = torch.cos(angle[:, 1::2])
    return pe.to(dtype=dtype)


class _ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: tuple[int, int]):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ConvEncoder(nn.Module):
    """Conv stem -> transformer -> mean-pool -> 384-dim L2-normalized embedding.

    Args:
        n_mels: number of mel bins (input freq dim).
        d_model: transformer hidden dim (also final conv channel count).
        n_layers: number of transformer layers.
        n_heads: attention heads.
        latent_dim: output embedding dim (384).
        return_raw: if True, also return the pre-normalization projection `h`
            (used for VICReg variance/covariance which needs unnormalized vectors).
    """

    def __init__(
        self,
        n_mels: int = 64,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        latent_dim: int = 384,
        ffn_ratio: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model

        # conv stem: 1x64xT -> 256 x (n_mels/8) x (T/8)
        self.stem = nn.Sequential(
            _ConvBlock(1, 48, (2, 2)),   # 32 x T/2
            _ConvBlock(48, 96, (2, 2)),  # 16 x T/4
            _ConvBlock(96, 192, (2, 2)),  # 8 x T/8
            _ConvBlock(192, d_model, (2, 1)),  # 4 x T/8
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))  # pool freq -> 1 x T/8

        # transformer over time
        self.pos_drop = nn.Dropout(0.1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ffn_ratio,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.project = nn.Sequential(
            nn.Linear(d_model, latent_dim),
            nn.BatchNorm1d(latent_dim),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, mel: torch.Tensor, mask: torch.Tensor | None = None,
                return_raw: bool = False):
        """mel: [B, 1, n_mels, T]. mask: optional [B, T] (True = keep) for input masking.

        Returns:
            if return_raw: (z, h) where z is L2-normalized [B, latent_dim],
                           h is raw projection [B, latent_dim].
            else: z [B, latent_dim].
        """
        B, _, Fm, T = mel.shape
        if mask is not None:
            # zero out masked time frames in the input spectrogram
            # mask: [B, T] with 1=keep, 0=mask
            mel = mel * mask[:, None, None, :]

        x = self.stem(mel)          # [B, d_model, F', T']
        x = self.freq_pool(x)       # [B, d_model, 1, T']
        x = x.squeeze(2)            # [B, d_model, T']
        x = x.transpose(1, 2)       # [B, T', d_model]

        pe = _sinusoidal_pos(self.d_model, x.shape[1], x.device, x.dtype)
        x = x + pe
        x = self.pos_drop(x)

        x = self.transformer(x)     # [B, T', d_model]
        x = x.mean(dim=1)           # [B, d_model]

        h = self.project(x)         # [B, latent_dim]
        z = F.normalize(h, dim=-1)

        if return_raw:
            return z, h
        return z


# --------------------------------------------------------------------------- #
#  Predictor
# --------------------------------------------------------------------------- #


class Predictor(nn.Module):
    """BYOL-style MLP predictor: maps context embedding to target embedding."""

    def __init__(self, dim: int = 384, hidden: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z), dim=-1)


# --------------------------------------------------------------------------- #
#  Full model (online encoder + predictor; target encoder managed externally)
# --------------------------------------------------------------------------- #


class AudioJEPA(nn.Module):
    """Online encoder + predictor. The EMA target encoder is a detached copy."""

    def __init__(self, encoder: ConvEncoder, predictor: Predictor):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor

    def forward(self, mel, mask=None, return_raw: bool = False):
        z, h = self.encoder(mel, mask=mask, return_raw=True)
        p = self.predictor(z)
        if return_raw:
            return p, z, h
        return p


def build_model(
    n_mels: int = 64,
    d_model: int = 256,
    n_layers: int = 2,
    n_heads: int = 4,
    latent_dim: int = 384,
    predictor_hidden: int = 768,
    device=None,
):
    """Build online encoder + predictor + a fresh (identical) target encoder."""
    encoder = ConvEncoder(
        n_mels=n_mels, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, latent_dim=latent_dim,
    )
    predictor = Predictor(dim=latent_dim, hidden=predictor_hidden)

    target_encoder = ConvEncoder(
        n_mels=n_mels, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, latent_dim=latent_dim,
    )
    # start the target as an exact copy of the online encoder
    target_encoder.load_state_dict(encoder.state_dict())
    for p in target_encoder.parameters():
        p.requires_grad_(False)

    model = AudioJEPA(encoder, predictor)
    if device is not None:
        model = model.to(device)
        target_encoder = target_encoder.to(device)
    return model, target_encoder


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
