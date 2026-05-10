# pipeline/preprocessor.py
# ============================================================
# Preprocessing citra X-ray
# Referensi: Ahmed et al. (2023), Diagnostics, 13(4), 814
# ============================================================

import cv2
import numpy as np
import logging
import io
import pydicom
from typing import Tuple

logger = logging.getLogger(__name__)


def load_image_from_bytes(img_bytes: bytes) -> np.ndarray:
    """
    Decode bytes dari upload HTTP → numpy grayscale uint8.
    Mendukung format PNG, JPG, dan DICOM.

    Args:
        img_bytes: raw bytes dari UploadFile.read()

    Returns:
        img_gray: numpy array (H, W), dtype uint8, grayscale

    Raises:
        ValueError: jika gambar gagal didecode
    """
    # 1. Coba load sebagai DICOM (Standard preamble 128 bytes + "DICM")
    try:
        # Pydicom butuh file-like object
        with io.BytesIO(img_bytes) as f:
            ds = pydicom.dcmread(f)
            # Ambil pixel data
            img = ds.pixel_array

            # Konversi ke float untuk normalisasi windowing
            img = img.astype(float)

            # Windowing / Rescaling (jika ada meta Rescale Slope/Intercept)
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                img = img * ds.RescaleSlope + ds.RescaleIntercept

            # Normalisasi ke 0-255 (grayscale uint8)
            img_min = img.min()
            img_max = img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min) * 255.0
            else:
                img = np.zeros_like(img)

            img = img.astype(np.uint8)

            # Handle Photometric Interpretation (Invert jika MONOCHROME1)
            # MONOCHROME1: 0 = white, MONOCHROME2: 0 = black (standar)
            if hasattr(ds, 'PhotometricInterpretation') and ds.PhotometricInterpretation == "MONOCHROME1":
                img = 255 - img

            logger.info("DICOM image detected and converted to grayscale uint8.")
            return img
    except Exception as e:
        # Jika bukan DICOM atau gagal, lanjut ke CV2
        logger.debug(f"Not a DICOM file or failed to read DICOM ({e}). Falling back to CV2.")

    # 2. Fallback ke OpenCV (PNG, JPG)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(
            "Gagal membaca gambar. Pastikan file adalah PNG, JPG, atau DICOM yang valid."
        )

    logger.debug(f"Image loaded via CV2: shape={img.shape}, dtype={img.dtype}")
    return img


def enhance_xray(img_gray: np.ndarray) -> np.ndarray:
    """
    Enhancement X-ray: Average Filter (5×5) + Laplacian subtraction.

    Pipeline:
    1. Average filter 5×5  → blur noise
    2. Laplacian filter     → deteksi tepi
    3. Subtract(avg, lap)   → perkuat tepi, kurangi noise

    Referensi: Ahmed et al. (2023), Diagnostics, 13(4), 814
    PENTING: Fungsi ini IDENTIK dengan notebook 03 dan 04 untuk konsistensi.

    Args:
        img_gray: numpy array (H, W), dtype uint8, grayscale

    Returns:
        enhanced: numpy array (H, W), dtype uint8
    """
    avg = cv2.blur(img_gray, (5, 5))
    lap = cv2.Laplacian(avg, cv2.CV_64F)
    lap = np.uint8(np.clip(np.absolute(lap), 0, 255))
    enhanced = cv2.subtract(avg, lap)

    logger.debug(f"Image enhanced: min={enhanced.min()}, max={enhanced.max()}")
    return enhanced


def preprocess_for_pipeline(
    img_bytes: bytes,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Entry point preprocessing: load + enhance.

    Args:
        img_bytes: raw bytes dari file upload

    Returns:
        img_original: numpy (H, W) uint8 — gambar grayscale asli (sebelum enhance)
        img_enhanced: numpy (H, W) uint8 — gambar setelah enhancement
    """
    img_original = load_image_from_bytes(img_bytes)
    img_enhanced = enhance_xray(img_original)

    return img_original, img_enhanced


def image_to_base64(img: np.ndarray, colormap: bool = False) -> str:
    """
    Konversi numpy array → base64 PNG string (untuk response JSON ke frontend).

    Args:
        img: numpy array (H, W) grayscale atau (H, W, 3) BGR
        colormap: jika True, apply COLORMAP_JET (untuk GradCAM heatmap)

    Returns:
        base64 string dengan prefix "data:image/png;base64,..."
    """
    import base64

    if colormap and len(img.shape) == 2:
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

    # Normalize ke 0-255 jika float
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

    success, buffer = cv2.imencode(".png", img)
    if not success:
        raise RuntimeError("Gagal encode gambar ke PNG")

    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64}"
