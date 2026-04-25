# pipeline/__init__.py
# ============================================================
# Orchestrator: menjalankan full pipeline dari bytes gambar
# sampai hasil akhir (prediksi + GradCAM + laporan)
#
# Flow:
# [1] load image bytes → grayscale
# [2] enhance_xray
# [3A] segmentasi → ROI (untuk handcraft)
# [3B] deep feature extraction (pre-GAP + post-GAP)
# [4] handcraft features dari ROI
# [5] PCA + fusion
# [6] ANN classify
# [7] GradCAM (hanya label terdeteksi)
# [8] LLM report
# ============================================================

import time
import logging
import traceback
import numpy as np
from typing import Dict, Any

from pipeline.preprocessor import preprocess_for_pipeline, image_to_base64
from pipeline.segmentation  import segment_and_get_roi
from pipeline.feature_handcraft import extract_handcrafted_features
from pipeline.feature_deep  import extract_deep_features
from pipeline.fusion        import fuse
from pipeline.classifier    import classify
from pipeline.gradcam       import generate_gradcam
from pipeline.reporter      import generate_report

logger = logging.getLogger(__name__)


def run_full_pipeline(img_bytes: bytes) -> Dict[str, Any]:
    """
    Jalankan full inference pipeline dari raw image bytes.

    Args:
        img_bytes: raw bytes dari HTTP upload

    Returns:
        result dict dengan keys:
        - predictions       : dict {label: bool}
        - probabilities     : dict {label: float}
        - gradcam_images    : dict {label: base64_png} — hanya label terdeteksi
        - original_image    : base64_png
        - roi_image         : base64_png
        - report            : str
        - processing_time_ms: int
        - pipeline_steps    : dict {step: time_ms} — untuk debugging

    Raises:
        Exception: jika pipeline gagal di tahap kritikal
    """
    t_total_start = time.perf_counter()
    step_times    = {}

    # ── [1] Load & Enhance ────────────────────────────────────
    t = time.perf_counter()
    img_original, img_enhanced = preprocess_for_pipeline(img_bytes)
    step_times["preprocess_ms"] = int((time.perf_counter() - t) * 1000)
    logger.info(f"[1] Preprocess: {step_times['preprocess_ms']}ms")

    # Simpan original untuk response (resize ke 512 max agar tidak terlalu besar)
    import cv2
    h, w = img_original.shape
    max_dim = 512
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_display = cv2.resize(img_original, (int(w * scale), int(h * scale)))
    else:
        img_display = img_original
    original_b64 = image_to_base64(img_display)

    # ── [2] Segmentasi + ROI ──────────────────────────────────
    t = time.perf_counter()
    try:
        roi, mask, coverage_pct = segment_and_get_roi(img_enhanced)
    except Exception as e:
        logger.error(f"Segmentasi gagal: {e}\n{traceback.format_exc()}")
        raise RuntimeError(f"Segmentasi ROI gagal: {e}") from e
    step_times["segmentation_ms"] = int((time.perf_counter() - t) * 1000)
    logger.info(
        f"[2] Segmentasi: {step_times['segmentation_ms']}ms, "
        f"coverage={coverage_pct:.1f}%"
    )

    roi_b64 = image_to_base64(roi)

    # ── [3] Handcraft Features ────────────────────────────────
    t = time.perf_counter()
    try:
        feat_hc = extract_handcrafted_features(roi)
    except Exception as e:
        logger.error(f"Handcraft extraction gagal: {e}")
        raise RuntimeError(f"Handcrafted feature extraction gagal: {e}") from e
    step_times["handcraft_ms"] = int((time.perf_counter() - t) * 1000)
    logger.info(
        f"[3] Handcraft: {step_times['handcraft_ms']}ms, "
        f"shape={feat_hc.shape}"
    )

    # ── [4] Deep Features ─────────────────────────────────────
    t = time.perf_counter()
    try:
        feat_post_gap, feat_pre_gap = extract_deep_features(img_enhanced)
    except Exception as e:
        logger.error(f"Deep feature extraction gagal: {e}")
        raise RuntimeError(f"Deep feature extraction gagal: {e}") from e
    step_times["deep_feature_ms"] = int((time.perf_counter() - t) * 1000)
    logger.info(
        f"[4] Deep feature: {step_times['deep_feature_ms']}ms, "
        f"post_gap={feat_post_gap.shape}"
    )

    # ── [5] PCA + Fusion ──────────────────────────────────────
    t = time.perf_counter()
    try:
        fused_vector = fuse(feat_post_gap, feat_hc)
    except Exception as e:
        logger.error(f"Fusion gagal: {e}")
        raise RuntimeError(f"PCA/Fusion gagal: {e}") from e
    step_times["fusion_ms"] = int((time.perf_counter() - t) * 1000)
    logger.info(
        f"[5] Fusion: {step_times['fusion_ms']}ms, "
        f"fused_dim={fused_vector.shape}"
    )

    # ── [6] ANN Classify ─────────────────────────────────────
    t = time.perf_counter()
    try:
        probabilities, predictions = classify(fused_vector)
    except Exception as e:
        logger.error(f"ANN classify gagal: {e}")
        raise RuntimeError(f"ANN classification gagal: {e}") from e
    step_times["classify_ms"] = int((time.perf_counter() - t) * 1000)
    logger.info(f"[6] Classify: {step_times['classify_ms']}ms")

    # ── [7] GradCAM ───────────────────────────────────────────
    t = time.perf_counter()
    try:
        gradcam_images = generate_gradcam(img_enhanced, predictions, probabilities)
    except Exception as e:
        logger.warning(f"GradCAM gagal (non-critical): {e}")
        gradcam_images = {}
    step_times["gradcam_ms"] = int((time.perf_counter() - t) * 1000)
    logger.info(
        f"[7] GradCAM: {step_times['gradcam_ms']}ms, "
        f"{len(gradcam_images)} images generated"
    )

    # ── [8] LLM Report ───────────────────────────────────────
    t = time.perf_counter()
    try:
        report = generate_report(probabilities, predictions, coverage_pct)
    except Exception as e:
        logger.warning(f"Report generation gagal (non-critical): {e}")
        report = "Laporan tidak tersedia. Silakan periksa konfigurasi API."
    step_times["report_ms"] = int((time.perf_counter() - t) * 1000)
    logger.info(f"[8] Report: {step_times['report_ms']}ms")

    # ── Total ─────────────────────────────────────────────────
    total_ms = int((time.perf_counter() - t_total_start) * 1000)
    logger.info(f"Pipeline selesai: {total_ms}ms total")

    return {
        "predictions"        : predictions,
        "probabilities"      : probabilities,
        "gradcam_images"     : gradcam_images,
        "original_image"     : original_b64,
        "roi_image"          : roi_b64,
        "report"             : report,
        "processing_time_ms" : total_ms,
        "pipeline_steps"     : step_times,
        "coverage_pct"       : round(coverage_pct, 1),
    }
