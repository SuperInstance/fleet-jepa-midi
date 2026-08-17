"""audio_jepa — the learned audio encoder (v2) for the fleet JEPA perception layer.

Replaces the hand-crafted ear in vibe_matcher.py with a self-supervised
Joint Embedding Predictive Architecture over log-mel spectrograms.

Pipeline:
    mel-spectrogram frontend -> conv/Conformer encoder -> 384-dim latent
    JEPA objective: masked-window embedding prediction with EMA target,
    stop-gradient, cosine-similarity predictor, and VICReg anti-collapse.
"""

from .model import MelFrontend, ConvEncoder, Predictor, AudioJEPA, build_model
from .dataset import SpeechClipsDataset, compute_mel

__all__ = [
    "MelFrontend",
    "ConvEncoder",
    "Predictor",
    "AudioJEPA",
    "build_model",
    "SpeechClipsDataset",
    "compute_mel",
]
