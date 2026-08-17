"""audio_jepa/dataset.py — tiny self-supervised audio dataset + augmentations.

Loads the 16 rendered speech clips once, precomputes their log-mel
spectrograms, caches them in RAM, and samples random (context, target) mel
windows. Because the corpus is tiny (16 clips), it is expanded with
mel-spectrogram-domain augmentations (SpecAugment-family):

    random gain     -> additive offset in dB
    background noise-> additive gaussian in dB
    pitch shift     -> vertical roll of mel bins (log-freq => pitch is translation)
    time-stretch /  -> resample of the time axis (tempo change)
    speed change

Mel-domain augmentation avoids per-sample STFT/pitch-shift/resample work
(~700 ms/sample on CPU), dropping it to well under 1 ms/sample so the model
trains in minutes. The same MelSpectrogram transform (via MelFrontend) is used
at inference so train/eval spectrograms are identical.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

AUDIO_SUFFIXES = (".mp3", ".wav", ".m4a", ".flac", ".ogg")


def discover_clips(corpus: str | Path) -> list[Path]:
    corpus = Path(corpus)
    clips = sorted(
        p for p in corpus.iterdir()
        if p.suffix.lower() in AUDIO_SUFFIXES and p.stat().st_size >= 50_000
    )
    return clips


def compute_mel(waveform: torch.Tensor, frontend, sample_rate: int = 16_000) -> torch.Tensor:
    """Convert a mono waveform [T] to a log-mel [n_mels, frames] via a MelFrontend."""
    w = waveform.reshape(1, 1, -1) if waveform.dim() == 1 else waveform
    return frontend(w)[0, 0]  # [n_mels, frames]


class SpeechClipsDataset:
    """Holds precomputed log-mels and yields augmented (context, target) mel windows."""

    def __init__(
        self,
        corpus: str | Path,
        frontend,
        sample_rate: int = 16_000,
        context_frames: int = 240,   # 2.4 s at 100 fps
        target_frames: int = 240,
        seed: int = 0,
    ):
        self.corpus = Path(corpus)
        self.sample_rate = sample_rate
        self.context_frames = context_frames
        self.target_frames = target_frames
        self.total_frames = context_frames + target_frames
        self.n_mels = frontend.n_mels

        import librosa
        self.clips = discover_clips(self.corpus)
        if not self.clips:
            raise RuntimeError(f"no audio clips found in {self.corpus}")

        self.mels: list[torch.Tensor] = []
        self.durations: list[float] = []
        for p in self.clips:
            y, sr = librosa.load(str(p), sr=sample_rate, mono=True)
            w = torch.from_numpy(y.astype(np.float32))
            m = compute_mel(w, frontend, sample_rate)
            self.mels.append(m)
            self.durations.append(w.numel() / sample_rate)

        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.clips)

    # --- mel-domain augmentations (fixed-length) --------------------------- #

    def _gain(self, m: torch.Tensor) -> torch.Tensor:
        db = self._rng.uniform(-6.0, 6.0)
        return m + db

    def _noise(self, m: torch.Tensor) -> torch.Tensor:
        sigma = self._rng.uniform(0.1, 1.5)
        return m + torch.randn_like(m) * sigma

    def _pitch_shift(self, m: torch.Tensor) -> torch.Tensor:
        # log-mel: a pitch shift ~ vertical translation; roll by a few bins
        bins = self._rng.randint(-3, 3)
        return torch.roll(m, shifts=bins, dims=0)

    def _stretch(self, m: torch.Tensor) -> torch.Tensor:
        # resample the time axis (tempo / speed change)
        r = self._rng.uniform(0.9, 1.1)
        T = m.shape[-1]
        new_T = max(1, int(round(T * r)))
        out = F.interpolate(m.unsqueeze(0), size=new_T, mode="linear",
                            align_corners=False).squeeze(0)
        if new_T > T:
            off = self._rng.randrange(0, new_T - T + 1)
            out = out[:, off:off + T]
        elif new_T < T:
            out = F.pad(out, (0, T - new_T))
        return out

    def augment(self, m: torch.Tensor) -> torch.Tensor:
        """Apply a random subset of mel-domain augmentations (keeps length)."""
        m = m.clone()
        m = self._gain(m)
        if self._rng.random() < 0.5:
            m = self._noise(m)
        r = self._rng.random()
        if r < 0.45:
            m = self._pitch_shift(m)
        elif r < 0.75:
            m = self._stretch(m)
        # else: no pitch/tempo change
        return m

    # --- sampling ---------------------------------------------------------- #

    def sample_window(self):
        """Return (ctx_mel [n_mels, ctx], tgt_mel [n_mels, tgt]) as tensors."""
        idx = self._rng.randrange(len(self.mels))
        m = self.mels[idx]
        total = self.total_frames
        if m.shape[-1] <= total:
            m = F.pad(m, (0, total - m.shape[-1] + 1))
        start = self._rng.randrange(0, m.shape[-1] - total)
        seg = m[:, start:start + total]
        seg = self.augment(seg)
        ctx = seg[:, : self.context_frames]
        tgt = seg[:, self.context_frames:]
        return ctx, tgt
