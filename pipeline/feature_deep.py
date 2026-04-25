import numpy as np
import torch
import torchvision
import torchxrayvision as xrv
import logging
from typing import Tuple, Dict

from app_config import (
    DEVICE,
    IMG_SIZE_HC,
    DEEP_FEATURE_DIM,
    GRADCAM_TARGET_LAYER,
)

logger = logging.getLogger(__name__)

# ── Singleton ─────────────────────────────────────────────────
_densenet_model = None


def load_deep_feature_model() -> None:
    """
    Load DenseNet121-XRV sekali saat startup.
    Download otomatis oleh torchxrayvision jika belum ada (~30 MB).
    """
    global _densenet_model

    logger.info("Loading DenseNet121-XRV model (weights=densenet121-res224-all)...")
    _densenet_model = xrv.models.DenseNet(weights="densenet121-res224-all")

    # Freeze semua parameter — hanya digunakan untuk feature extraction
    for param in _densenet_model.parameters():
        param.requires_grad = False

    _densenet_model = _densenet_model.to(DEVICE)
    _densenet_model.eval()

    # Verifikasi output dimension
    with torch.no_grad():
        dummy = torch.zeros(1, 1, IMG_SIZE_HC, IMG_SIZE_HC).to(DEVICE)
        feat  = _densenet_model.features(dummy)
        feat_gap = feat.mean([-2, -1])
        assert feat_gap.shape[1] == DEEP_FEATURE_DIM, (
            f"Feature dim salah: {feat_gap.shape[1]} (harusnya {DEEP_FEATURE_DIM})"
        )

    logger.info(
        f"DenseNet121-XRV loaded. Output dim: {DEEP_FEATURE_DIM} (after GAP)"
    )


def _get_deep_model():
    """Ambil model yang sudah di-load."""
    if _densenet_model is None:
        raise RuntimeError(
            "Deep feature model belum di-load. Panggil load_deep_feature_model() dulu."
        )
    return _densenet_model


def _preprocess_for_densenet(img_enhanced: np.ndarray) -> torch.Tensor:
    """
    Preprocess untuk DenseNet121-XRV sesuai dokumentasi XRV.

    Pipeline:
    1. normalize ke [-1024, 1024]  (tanpa reshape karena sudah 2D)
    2. tambah channel dim → (1, H, W)
    3. XRayCenterCrop + XRayResizer(224)
    4. Konversi ke tensor float32

    Args:
        img_enhanced: numpy (H, W) uint8

    Returns:
        tensor: (1, 1, 224, 224) di DEVICE
    """
    # Normalize sesuai XRV — tanpa reshape
    img_norm = xrv.datasets.normalize(img_enhanced, 255)  # float32, [-1024, 1024]

    # Tambah channel dim
    img_norm = img_norm[None, ...]  # (1, H, W)

    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(IMG_SIZE_HC),
    ])
    img_224 = transform(img_norm)   # (1, 224, 224)

    tensor = torch.from_numpy(img_224).unsqueeze(0).float().to(DEVICE)  # (1, 1, 224, 224)
    return tensor


def extract_deep_features(
    img_enhanced: np.ndarray,
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Ekstrak deep features dari gambar enhanced.

    Args:
        img_enhanced: numpy (H, W) uint8, output dari preprocessor.enhance_xray()

    Returns:
        feat_post_gap : numpy (1024,) float32   → untuk PCA + Fusion
        feat_pre_gap  : torch.Tensor (1, 1024, 7, 7) → untuk GradCAM
    """
    model  = _get_deep_model()
    tensor = _preprocess_for_densenet(img_enhanced)

    # ── Forward hook untuk capture pre-GAP feature map ────────
    pre_gap_store: Dict[str, torch.Tensor] = {}

    def hook_fn(module, input, output):
        # output dari densenet121.features: (B, 1024, 7, 7)
        pre_gap_store["feat"] = output.detach().clone()

    # Register hook pada layer features (output sebelum classifier/GAP)
    hook = model.densenet121.features.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            _ = model.features(tensor)   # trigger hook, output adalah post-GAP (1, 1024)
    finally:
        hook.remove()

    # pre-GAP: dari hook (1, 1024, 7, 7) — simpan untuk GradCAM
    feat_pre_gap = pre_gap_store["feat"]   # Tensor (1, 1024, 7, 7) di DEVICE

    # post-GAP: Global Average Pool manual → (1024,)
    feat_post_gap = feat_pre_gap.mean(dim=[-2, -1]).squeeze(0).cpu().numpy()  # (1024,)
    feat_post_gap = feat_post_gap.astype(np.float32)

    # Validasi
    if np.isnan(feat_post_gap).any() or np.isinf(feat_post_gap).any():
        logger.warning("Deep feature mengandung NaN/Inf, normalisasi ulang")
        feat_post_gap = np.nan_to_num(feat_post_gap, nan=0.0, posinf=1e6, neginf=-1e6)

    logger.debug(
        f"Deep features extracted: post_gap={feat_post_gap.shape}, "
        f"pre_gap={feat_pre_gap.shape}"
    )

    return feat_post_gap, feat_pre_gap


def get_densenet_model():
    """
    Expose model untuk keperluan GradCAM.
    Model dikembalikan dalam mode eval, freeze.
    """
    return _get_deep_model()


def get_target_layer():
    """
    Kembalikan layer target GradCAM (densenet121.features.denseblock4).
    Layer ini adalah output terakhir DenseBlock sebelum normalization dan GAP.
    """
    model = _get_deep_model()

    # Akses nested layer: densenet121.features.denseblock4
    try:
        target_layer = model.densenet121.features.denseblock4
    except AttributeError:
        # Fallback: coba akses via nama string
        parts = GRADCAM_TARGET_LAYER.split(".")
        layer = model
        for part in parts:
            layer = getattr(layer, part)
        target_layer = layer

    return target_layer
