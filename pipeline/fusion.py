# pipeline/fusion.py
# ============================================================
# PCA Transform + Feature Fusion
# Load scaler_deep, pca_deep, scaler_hc dari file pkl (hasil training)
# Identik dengan logika notebook 05
# ============================================================

import numpy as np
import joblib
import logging
from pathlib import Path

from app_config import (
    PCA_DEEP_PATH,
    SCALER_DEEP_PATH,
    SCALER_HC_PATH,
    HC_FEATURE_DIM,
    DEEP_FEATURE_DIM,
)

logger = logging.getLogger(__name__)

# ── Singleton ─────────────────────────────────────────────────
_scaler_deep = None
_pca_deep    = None
_scaler_hc   = None
_fused_dim   = None   # dimensi setelah fusion (PCA_dim + 35)


def load_fusion_artifacts() -> None:
    """
    Load scaler dan PCA objects dari file pkl.
    Dipanggil sekali saat startup FastAPI.

    Files yang diperlukan (letakkan di backend/models/):
    - scaler_deep.pkl  (dari 05_pca_fusion/)
    - pca_deep.pkl     (dari 05_pca_fusion/)
    - scaler_hc.pkl    (dari 05_pca_fusion/)
    """
    global _scaler_deep, _pca_deep, _scaler_hc, _fused_dim

    # Validasi file ada
    for path, name in [
        (SCALER_DEEP_PATH, "scaler_deep.pkl"),
        (PCA_DEEP_PATH,    "pca_deep.pkl"),
        (SCALER_HC_PATH,   "scaler_hc.pkl"),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"File model tidak ditemukan: {path}\n"
                f"Pastikan '{name}' sudah di-copy ke folder backend/models/"
            )

    logger.info("Loading PCA dan Scaler artifacts...")

    _scaler_deep = joblib.load(SCALER_DEEP_PATH)
    _pca_deep    = joblib.load(PCA_DEEP_PATH)
    _scaler_hc   = joblib.load(SCALER_HC_PATH)

    # Hitung dimensi setelah PCA
    n_pca      = _pca_deep.n_components_
    _fused_dim = n_pca + HC_FEATURE_DIM

    logger.info(
        f"Fusion artifacts loaded.\n"
        f"  PCA components   : {n_pca}\n"
        f"  HC features      : {HC_FEATURE_DIM}\n"
        f"  Fused dim total  : {_fused_dim}"
    )


def _check_loaded():
    if _scaler_deep is None or _pca_deep is None or _scaler_hc is None:
        raise RuntimeError(
            "Fusion artifacts belum di-load. Panggil load_fusion_artifacts() dulu."
        )


def fuse(
    feat_deep_post_gap: np.ndarray,
    feat_hc: np.ndarray,
) -> np.ndarray:
    """
    Gabungkan deep features (setelah PCA) + handcrafted features (setelah scaling).

    Pipeline:
    1. scaler_deep.transform(feat_1024)  → deep features scaled
    2. pca_deep.transform(deep_scaled)   → deep features PCA reduced (N_pca dim)
    3. scaler_hc.transform(feat_35)      → HC features scaled
    4. concatenate([pca_feats, hc_scaled]) → fused vector (N_pca + 35 dim)

    Args:
        feat_deep_post_gap : numpy (1024,) — output dari feature_deep.extract_deep_features()
        feat_hc            : numpy (35,)   — output dari feature_handcraft.extract_handcrafted_features()

    Returns:
        fused_vector: numpy (N_pca + 35,) float32
    """
    _check_loaded()

    # Reshape ke (1, dim) untuk sklearn transform
    deep_reshaped = feat_deep_post_gap.reshape(1, -1).astype(np.float64)
    hc_reshaped   = feat_hc.reshape(1, -1).astype(np.float64)

    # Step 1 & 2: Scale + PCA deep features
    deep_scaled = _scaler_deep.transform(deep_reshaped)       # (1, 1024)
    deep_pca    = _pca_deep.transform(deep_scaled)            # (1, N_pca)

    # Step 3: Scale HC features
    hc_scaled = _scaler_hc.transform(hc_reshaped)             # (1, 35)

    # Step 4: Concatenate
    fused = np.concatenate([deep_pca, hc_scaled], axis=1)     # (1, N_pca+35)

    fused_vector = fused.squeeze(0).astype(np.float32)        # (N_pca+35,)

    logger.debug(f"Fused feature shape: {fused_vector.shape}")
    return fused_vector


def get_fused_dim() -> int:
    """Kembalikan dimensi fused feature vector (untuk validasi ANN input_dim)."""
    _check_loaded()
    return _fused_dim
