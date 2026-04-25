# pipeline/segmentation.py
# ============================================================
# Segmentasi paru-paru menggunakan PSPNet dari TorchXRayVision
# Output: binary mask + ROI (gambar yang di-mask)
# Referensi: TorchXRayVision (Cohen et al., 2020)
# ============================================================

import cv2
import numpy as np
import torch
import torchvision
import torchxrayvision as xrv
import logging
from typing import Tuple

from app_config import (
    DEVICE,
    IMG_SIZE_HC,
    IMG_SIZE_SEG,
    MASK_THRESHOLD,
    MORPH_KERNEL_SIZE,
)

logger = logging.getLogger(__name__)

# ── Singleton: model di-load sekali saat startup ──────────────
_seg_model = None
_left_idx  = None
_right_idx = None


def load_segmentation_model() -> None:
    """
    Load PSPNet sekali saat startup FastAPI.
    Download otomatis oleh torchxrayvision jika belum ada.
    """
    global _seg_model, _left_idx, _right_idx

    logger.info("Loading PSPNet segmentation model...")
    _seg_model = xrv.baseline_models.chestx_det.PSPNet()
    _seg_model = _seg_model.to(DEVICE)
    _seg_model.eval()

    # Index channel untuk Left Lung dan Right Lung
    _left_idx  = _seg_model.targets.index("Left Lung")
    _right_idx = _seg_model.targets.index("Right Lung")

    logger.info(
        f"PSPNet loaded. Left Lung idx={_left_idx}, Right Lung idx={_right_idx}"
    )


def _get_seg_model():
    """Ambil model yang sudah di-load (raise jika belum)."""
    if _seg_model is None:
        raise RuntimeError(
            "Segmentation model belum di-load. Panggil load_segmentation_model() dulu."
        )
    return _seg_model, _left_idx, _right_idx


def _preprocess_for_seg(img_enhanced: np.ndarray) -> Tuple[torch.Tensor, tuple]:
    """
    Preprocess gambar untuk input PSPNet.

    Pipeline sesuai dokumentasi XRV:
    1. Normalize ke [-1024, 1024]
    2. XRayCenterCrop + XRayResizer(512)
    3. Bungkus jadi tensor (1, 1, 512, 512)

    Args:
        img_enhanced: numpy (H, W) uint8, sudah di-enhance

    Returns:
        tensor  : torch.Tensor (1, 1, 512, 512) di DEVICE
        orig_size: (H, W) ukuran asli sebelum resize
    """
    orig_size = img_enhanced.shape[:2]  # (H, W)

    # Normalize ke [-1024, 1024] sesuai XRV — reshape=True menambah channel dim
    img_norm = xrv.datasets.normalize(img_enhanced, maxval=255, reshape=True)
    # Shape sekarang: (1, H, W)

    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(IMG_SIZE_SEG),
    ])
    img_512 = transform(img_norm)  # (1, 512, 512)

    tensor = torch.from_numpy(img_512).unsqueeze(0).to(DEVICE)  # (1, 1, 512, 512)
    return tensor, orig_size


def _postprocess_mask(
    model_output: torch.Tensor,
    orig_size: tuple,
    left_idx: int,
    right_idx: int,
    threshold: float = MASK_THRESHOLD,
) -> np.ndarray:
    """
    Konversi output PSPNet → binary mask ukuran asli.

    Steps:
    1. Ambil channel Left Lung & Right Lung
    2. Union dengan np.maximum
    3. Threshold → binary (0 atau 255)
    4. Resize ke orig_size dengan INTER_NEAREST

    Args:
        model_output: tensor (1, 14, 512, 512)
        orig_size   : (H, W) ukuran gambar asli
        threshold   : ambang probabilitas

    Returns:
        mask_binary: numpy (H, W) uint8, nilai 0 atau 255
    """
    left_prob  = model_output[0, left_idx].cpu().numpy()   # (512, 512)
    right_prob = model_output[0, right_idx].cpu().numpy()  # (512, 512)

    combined = np.maximum(left_prob, right_prob)            # union
    mask_512 = (combined >= threshold).astype(np.uint8) * 255

    mask_orig = cv2.resize(
        mask_512,
        (orig_size[1], orig_size[0]),   # (W, H) untuk cv2
        interpolation=cv2.INTER_NEAREST,
    )
    return mask_orig


def _morphological_cleanup(mask: np.ndarray, kernel_size: int = MORPH_KERNEL_SIZE) -> np.ndarray:
    """
    Bersihkan mask dari noise:
    - MORPH_CLOSE: tutup lubang kecil di dalam paru
    - MORPH_OPEN : hilangkan bercak kecil di luar paru
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    return mask


def segment_and_get_roi(
    img_enhanced: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Pipeline segmentasi lengkap: enhanced image → mask → ROI.

    Args:
        img_enhanced: numpy (H, W) uint8, sudah di-enhance oleh preprocessor

    Returns:
        roi       : numpy (224, 224) uint8 — gambar ter-mask, resize ke HC size
        mask_vis  : numpy (H, W) uint8 — binary mask (0/255) ukuran asli, untuk visualisasi
        coverage  : float — persentase area paru (0.0–100.0)
    """
    model, left_idx, right_idx = _get_seg_model()

    # Preprocess untuk PSPNet
    tensor, orig_size = _preprocess_for_seg(img_enhanced)

    # Inferensi PSPNet
    with torch.no_grad():
        output = model(tensor)  # (1, 14, 512, 512)

    # Postprocess → binary mask ukuran asli
    mask = _postprocess_mask(output, orig_size, left_idx, right_idx)
    mask = _morphological_cleanup(mask)

    # Coverage (% area paru)
    coverage = float((mask > 0).sum() / mask.size * 100)
    logger.debug(f"Lung coverage: {coverage:.1f}%")

    # Buat ROI: apply mask pada gambar enhanced, resize ke 224×224
    img_resized  = cv2.resize(img_enhanced, (IMG_SIZE_HC, IMG_SIZE_HC))
    mask_resized = cv2.resize(
        mask, (IMG_SIZE_HC, IMG_SIZE_HC), interpolation=cv2.INTER_NEAREST
    )
    _, mask_bin = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)
    roi = cv2.bitwise_and(img_resized, img_resized, mask=mask_bin)

    return roi, mask, coverage
