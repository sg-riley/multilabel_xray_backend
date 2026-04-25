# pipeline/feature_handcraft.py
# ============================================================
# Ekstraksi Handcrafted Feature dari ROI
# GLCM (13 fitur) + LBP (10 fitur) + DWT (12 fitur) = 35 fitur
# Identik dengan notebook 03 (process_one_image)
# ============================================================

import numpy as np
import pywt
import logging
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

from app_config import HC_FEATURE_DIM

logger = logging.getLogger(__name__)

# Nama fitur untuk referensi / debugging
FEATURE_NAMES = (
    [
        "glcm_contrast",       "glcm_dissimilarity",  "glcm_homogeneity",
        "glcm_energy",         "glcm_correlation",    "glcm_ASM",
        "glcm_mean",           "glcm_variance",       "glcm_std",
        "glcm_entropy",        "glcm_max_prob",
        "glcm_cluster_shade",  "glcm_cluster_prom",
    ]
    + [f"lbp_{i}" for i in range(10)]
    + [
        "dwt_LL_mean", "dwt_LL_var", "dwt_LL_std",
        "dwt_LH_mean", "dwt_LH_var", "dwt_LH_std",
        "dwt_HL_mean", "dwt_HL_var", "dwt_HL_std",
        "dwt_HH_mean", "dwt_HH_var", "dwt_HH_std",
    ]
)

assert len(FEATURE_NAMES) == HC_FEATURE_DIM, (
    f"FEATURE_NAMES count mismatch: {len(FEATURE_NAMES)} != {HC_FEATURE_DIM}"
)


def _extract_glcm(roi: np.ndarray) -> np.ndarray:
    """
    Ekstrak 13 fitur GLCM.

    Config:
    - distances=[1], angles=[0°, 45°, 90°, 135°]
    - levels=16, symmetric=True, normed=True
    - 6 fitur via graycoprops + 7 fitur statistik manual

    Returns:
        np.ndarray shape (13,)
    """
    distances = [1]
    angles    = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    # Quantize ke 16 level
    roi_scaled = (roi / 16).astype(np.uint8)

    glcm = graycomatrix(
        roi_scaled,
        distances=distances,
        angles=angles,
        levels=16,
        symmetric=True,
        normed=True,
    )

    # 6 fitur dari graycoprops
    features = []
    for prop in ["contrast", "dissimilarity", "homogeneity",
                 "energy", "correlation", "ASM"]:
        features.append(float(graycoprops(glcm, prop).mean()))

    # 7 fitur statistik manual
    glcm_mean = glcm[:, :, 0, :].mean(axis=2)   # rata-rata 4 sudut → (16, 16)
    levels    = np.arange(16)
    i_idx, j_idx = np.meshgrid(levels, levels, indexing="ij")

    p     = glcm_mean
    mu_i  = float((i_idx * p).sum())
    mu_j  = float((j_idx * p).sum())
    var_i = float((((i_idx - mu_i) ** 2) * p).sum())

    entropy         = float(-(p * np.log2(p + 1e-10)).sum())
    max_prob        = float(p.max())
    cluster_shade   = float((((i_idx + j_idx - mu_i - mu_j) ** 3) * p).sum())
    cluster_prom    = float((((i_idx + j_idx - mu_i - mu_j) ** 4) * p).sum())

    features.extend([
        mu_i,           # mean
        var_i,          # variance
        float(np.sqrt(var_i)),  # std
        entropy,
        max_prob,
        cluster_shade,
        cluster_prom,
    ])

    return np.array(features, dtype=np.float32)   # (13,)


def _extract_lbp(roi: np.ndarray, P: int = 8, R: int = 1) -> np.ndarray:
    """
    Ekstrak 10 fitur LBP (P=8, R=1, method='uniform').

    n_bins = P + 2 = 10 (ada P+2 pola uniform yang mungkin)

    Returns:
        np.ndarray shape (10,)
    """
    lbp    = local_binary_pattern(roi, P=P, R=R, method="uniform")
    n_bins = P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)   # (10,)


def _extract_dwt(roi: np.ndarray, wavelet: str = "db1") -> np.ndarray:
    """
    Ekstrak 12 fitur DWT (1-level 2D DWT, wavelet=db1).

    Sub-bands: LL, LH, HL, HH
    Fitur per sub-band: mean, variance, std → 4 × 3 = 12

    Returns:
        np.ndarray shape (12,)
    """
    coeffs2 = pywt.dwt2(roi.astype(np.float32), wavelet)
    LL, (LH, HL, HH) = coeffs2

    features = []
    for subband in [LL, LH, HL, HH]:
        features.extend([
            float(np.mean(subband)),
            float(np.var(subband)),
            float(np.std(subband)),
        ])

    return np.array(features, dtype=np.float32)   # (12,)


def extract_handcrafted_features(roi: np.ndarray) -> np.ndarray:
    """
    Ekstrak semua 35 handcrafted features dari ROI.

    Args:
        roi: numpy (224, 224) uint8 — output dari segmentation.segment_and_get_roi()

    Returns:
        features: numpy (35,) float32

    Raises:
        ValueError: jika ROI invalid atau feature extraction gagal
    """
    if roi is None or roi.size == 0:
        raise ValueError("ROI kosong — tidak bisa ekstrak handcrafted features")

    if roi.shape != (224, 224):
        logger.warning(f"ROI shape tidak standar: {roi.shape}, akan di-resize")
        import cv2
        roi = cv2.resize(roi, (224, 224))

    glcm_feats = _extract_glcm(roi)     # (13,)
    lbp_feats  = _extract_lbp(roi)      # (10,)
    dwt_feats  = _extract_dwt(roi)      # (12,)

    features = np.concatenate([glcm_feats, lbp_feats, dwt_feats])  # (35,)

    # Validasi
    if np.isnan(features).any():
        logger.warning("Ada NaN di handcrafted features, diganti 0")
        features = np.nan_to_num(features, nan=0.0)

    if np.isinf(features).any():
        logger.warning("Ada Inf di handcrafted features, di-clip")
        features = np.clip(features, -1e9, 1e9)

    assert features.shape == (HC_FEATURE_DIM,), (
        f"Feature shape salah: {features.shape} (harusnya ({HC_FEATURE_DIM},))"
    )

    logger.debug(f"Handcrafted features extracted: shape={features.shape}")
    return features
