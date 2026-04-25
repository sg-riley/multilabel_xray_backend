import numpy as np
import torch
import torch.nn as nn
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

from app_config import (
    ANN_MODEL_PATH,
    OPTIMAL_THRESHOLDS_PATH,
    DEVICE,
    TARGET_LABELS,
    N_LABELS,
)

logger = logging.getLogger(__name__)


# ── Definisi ANN ─────────────────
class ANNClassifier(nn.Module):
    """
    ANN Classifier multi-label.

    Arsitektur:
        Dense(128) → BN → ReLU → Dropout(0.4)
        Dense(64)  → BN → ReLU → Dropout(0.3)
        Dense(32)  → BN → ReLU → Dropout(0.2)
        Dense(N_LABELS) → raw logits (NO Sigmoid di sini)

    Output: raw logits — Sigmoid diterapkan saat inferensi.
    """

    def __init__(self, input_dim: int, n_labels: int = N_LABELS):
        super(ANNClassifier, self).__init__()

        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            # Layer 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            # Layer 3
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            # Output — raw logits
            nn.Linear(32, n_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ── Singleton ─────────────────────────────────────────────────
_ann_model          = None
_optimal_thresholds = None
_input_dim          = None


def load_classifier() -> None:
    """
    Load ANN model + optimal thresholds dari file.
    Dipanggil sekali saat startup FastAPI.

    Files yang diperlukan (di backend/models/):
    - ann_best.pt             (rename dari ann_densenet121_xrv_all_aug_best.pt)
    - optimal_thresholds.json (rename dari optimal_thresholds_densenet121_xrv_all_aug.json)
    """
    global _ann_model, _optimal_thresholds, _input_dim

    # Validasi file
    for path, name in [
        (ANN_MODEL_PATH,          "ann_best.pt"),
        (OPTIMAL_THRESHOLDS_PATH, "optimal_thresholds.json"),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"File model tidak ditemukan: {path}\n"
                f"Pastikan '{name}' sudah di-copy ke folder backend/models/"
            )

    # Load checkpoint
    logger.info("Loading ANN classifier...")
    checkpoint = torch.load(ANN_MODEL_PATH, map_location=DEVICE)

    _input_dim = checkpoint["input_dim"]
    n_labels   = checkpoint.get("n_labels", N_LABELS)

    logger.info(
        f"ANN checkpoint loaded.\n"
        f"  Best epoch  : {checkpoint.get('epoch', '?')}\n"
        f"  Val loss    : {checkpoint.get('val_loss', '?'):.4f}\n"
        f"  Input dim   : {_input_dim}\n"
        f"  N labels    : {n_labels}"
    )

    # Instantiate dan load weights
    _ann_model = ANNClassifier(_input_dim, n_labels).to(DEVICE)
    _ann_model.load_state_dict(checkpoint["model_state"])
    _ann_model.eval()

    # Load optimal thresholds
    with open(OPTIMAL_THRESHOLDS_PATH, "r") as f:
        _optimal_thresholds = json.load(f)

    logger.info(
        f"Optimal thresholds loaded: "
        + ", ".join(f"{k}={v:.3f}" for k, v in _optimal_thresholds.items())
    )


def _check_loaded():
    if _ann_model is None or _optimal_thresholds is None:
        raise RuntimeError(
            "Classifier belum di-load. Panggil load_classifier() dulu."
        )


def classify(
    fused_vector: np.ndarray,
) -> Tuple[Dict[str, float], Dict[str, bool]]:
    """
    Jalankan inferensi ANN pada fused feature vector.

    Pipeline:
    1. numpy → tensor
    2. forward pass → raw logits
    3. sigmoid → probabilities
    4. prob >= threshold[label] → binary prediction

    Args:
        fused_vector: numpy (N_pca + 35,) float32

    Returns:
        probabilities: dict {label: float (0.0–1.0)}
        predictions  : dict {label: bool}

    Raises:
        ValueError: jika dimensi fused_vector tidak cocok dengan model
    """
    _check_loaded()

    # Validasi dimensi
    if fused_vector.shape[0] != _input_dim:
        raise ValueError(
            f"Dimensi fused_vector tidak sesuai: "
            f"dapat {fused_vector.shape[0]}, model ekspektasi {_input_dim}"
        )

    # Tensor (1, input_dim)
    x = torch.from_numpy(fused_vector).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = _ann_model(x)                          # (1, N_labels)
        probs  = torch.sigmoid(logits).squeeze(0)       # (N_labels,)

    probs_np = probs.cpu().numpy()

    # Buat dict probabilities dan predictions
    probabilities = {}
    predictions   = {}

    for i, label in enumerate(TARGET_LABELS):
        prob      = float(probs_np[i])
        threshold = _optimal_thresholds.get(label, 0.5)

        probabilities[label] = round(prob, 4)
        predictions[label]   = bool(prob >= threshold)

    logger.debug(
        "Predictions: "
        + ", ".join(f"{k}={'✓' if v else '✗'}" for k, v in predictions.items())
    )

    return probabilities, predictions


def get_input_dim() -> int:
    """Kembalikan input dimensi ANN (untuk cross-check dengan fusion)."""
    _check_loaded()
    return _input_dim
