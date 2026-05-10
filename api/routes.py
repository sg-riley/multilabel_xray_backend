# api/routes.py
# ============================================================
# FastAPI route definitions
# ============================================================

import logging
import traceback
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional

from pipeline import run_full_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["xray"])


# ── Response Models ───────────────────────────────────────────

class AnalysisResponse(BaseModel):
    predictions         : Dict[str, bool]
    probabilities       : Dict[str, float]
    gradcam_images      : Dict[str, str]    # label → base64 PNG (hanya yang terdeteksi)
    original_image      : str               # base64 PNG
    roi_image           : str               # base64 PNG
    report              : str
    processing_time_ms  : int
    pipeline_steps      : Dict[str, int]    # step timing breakdown
    coverage_pct        : float             # % area paru terdeteksi


class HealthResponse(BaseModel):
    status  : str
    message : str
    models  : Dict[str, bool]


# ── Endpoints ─────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Cek apakah semua model sudah ter-load dengan benar.
    """
    from pipeline.segmentation  import _seg_model
    from pipeline.feature_deep  import _densenet_model
    from pipeline.classifier    import _ann_model
    from pipeline.fusion        import _pca_deep

    models_status = {
        "pspnet_segmentation" : _seg_model is not None,
        "densenet121_xrv"     : _densenet_model is not None,
        "ann_classifier"      : _ann_model is not None,
        "pca_scaler"          : _pca_deep is not None,
    }

    all_ready = all(models_status.values())

    return HealthResponse(
        status  = "ready" if all_ready else "loading",
        message = "Semua model siap" if all_ready else "Beberapa model belum ter-load",
        models  = models_status,
    )


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_xray(
    file: UploadFile = File(
        ...,
        description="File citra X-ray (PNG atau JPG, max 10 MB)",
    )
):
    """
    Endpoint utama: analisis citra X-ray.

    Pipeline:
    1. Preprocessing (enhance)
    2. Segmentasi paru (PSPNet) → ROI
    3. Handcrafted features (GLCM + LBP + DWT) dari ROI
    4. Deep features (DenseNet121-XRV) dari gambar asli
    5. PCA + Fusion
    6. Klasifikasi multi-label (ANN)
    7. GradCAM (untuk label terdeteksi)
    8. Laporan LLM

    Returns:
        JSON dengan prediksi, probabilitas, GradCAM, dan laporan
    """
    # Validasi file type
    allowed_types = ["image/png", "image/jpeg", "image/jpg", "application/dicom", "application/octet-stream"]
    allowed_exts  = (".png", ".jpg", ".jpeg", ".dcm", ".dicom")

    is_valid_type = file.content_type in allowed_types
    is_valid_ext  = file.filename.lower().endswith(allowed_exts)

    if not (is_valid_type or is_valid_ext):
        raise HTTPException(
            status_code=422,
            detail=f"Format file tidak didukung: {file.content_type}. Gunakan PNG, JPG, atau DICOM.",
        )

    # Baca bytes
    img_bytes = await file.read()

    # Validasi ukuran (max 10 MB)
    max_size = 10 * 1024 * 1024
    if len(img_bytes) > max_size:
        raise HTTPException(
            status_code=422,
            detail=f"Ukuran file terlalu besar: {len(img_bytes)/1e6:.1f} MB (max 10 MB)",
        )

    if len(img_bytes) == 0:
        raise HTTPException(status_code=422, detail="File kosong")

    logger.info(
        f"Menerima request: file={file.filename}, "
        f"size={len(img_bytes)/1024:.1f}KB, "
        f"type={file.content_type}"
    )

    # Jalankan pipeline
    try:
        result = run_full_pipeline(img_bytes)
    except ValueError as e:
        # Error input (mis. gambar tidak valid)
        logger.warning(f"Input error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        # Error pipeline (mis. model gagal)
        logger.error(f"Pipeline error: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {e}",
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail="Terjadi kesalahan internal. Cek log server.",
        )

    return result


@router.post("/preview")
async def get_image_preview(
    file: UploadFile = File(...)
):
    """
    Generate preview PNG (base64) untuk file gambar/DICOM.
    Digunakan oleh frontend untuk menampilkan preview DICOM.
    """
    img_bytes = await file.read()
    
    try:
        from pipeline.preprocessor import load_image_from_bytes, image_to_base64
        img = load_image_from_bytes(img_bytes)
        # Resize untuk preview agar tidak terlalu berat (opsional, tapi bagus)
        # img = cv2.resize(img, (512, 512)) 
        return {"preview": image_to_base64(img)}
    except Exception as e:
        logger.error(f"Preview error: {e}")
        raise HTTPException(status_code=422, detail="Gagal membuat preview gambar")


@router.get("/labels")
async def get_labels():
    """Kembalikan daftar target labels yang diclassify."""
    from app_config import TARGET_LABELS
    return {"labels": TARGET_LABELS}
