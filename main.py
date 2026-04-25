# main.py
# ============================================================
# FastAPI Entry Point — Hybrid CNN-ANN Chest X-Ray Classifier
#
# Startup sequence (lifespan):
# 1. Set torch threads untuk CPU optimization
# 2. Load PSPNet (segmentasi)
# 3. Load DenseNet121-XRV (deep feature)
# 4. Load Fusion artifacts (scaler + PCA)
# 5. Load ANN classifier + optimal thresholds
#
# Semua model di-load SEKALI ke RAM saat startup,
# lalu digunakan untuk setiap request (singleton pattern).
# ============================================================

import logging
import sys
import time
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app_config import TORCH_NUM_THREADS, ALLOWED_ORIGINS
from api.routes import router

# ── Logging Setup ─────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt = "%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ── Model Loading Functions ───────────────────────────────────

def load_all_models():
    """
    Load semua model ke memori secara berurutan.
    Dipanggil saat FastAPI startup via lifespan context.
    """
    from pipeline.segmentation  import load_segmentation_model
    from pipeline.feature_deep  import load_deep_feature_model
    from pipeline.fusion        import load_fusion_artifacts
    from pipeline.classifier    import load_classifier, get_input_dim
    from pipeline.fusion        import get_fused_dim

    # Optimasi CPU
    torch.set_num_threads(TORCH_NUM_THREADS)
    logger.info(f"PyTorch CPU threads: {TORCH_NUM_THREADS}")

    steps = [
        ("PSPNet Segmentation",       load_segmentation_model),
        ("DenseNet121-XRV",           load_deep_feature_model),
        ("PCA + Scaler (Fusion)",     load_fusion_artifacts),
        ("ANN Classifier + Threshold", load_classifier),
    ]

    total_start = time.time()
    for name, fn in steps:
        t = time.time()
        logger.info(f"Loading: {name}...")
        fn()
        logger.info(f"✓ {name} loaded in {(time.time()-t):.1f}s")

    # Cross-check dimensi ANN vs Fusion
    ann_input = get_input_dim()
    fused_dim = get_fused_dim()
    if ann_input != fused_dim:
        logger.error(
            f"DIMENSI MISMATCH: ANN expects {ann_input} "
            f"but fusion produces {fused_dim}!\n"
            "Pastikan model ann_best.pt dan pkl files berasal dari "
            "training yang sama."
        )
        raise ValueError(
            f"Input dimension mismatch: ANN={ann_input}, Fusion={fused_dim}"
        )
    else:
        logger.info(f"✓ Dimension check passed: {ann_input} == {fused_dim}")

    elapsed = time.time() - total_start
    logger.info(
        f"\n{'='*50}\n"
        f"  SEMUA MODEL SIAP ({elapsed:.1f}s)\n"
        f"  Fused feature dim : {fused_dim}\n"
        f"  ANN input dim     : {ann_input}\n"
        f"  Device            : CPU\n"
        f"{'='*50}"
    )


# ── Lifespan ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Load models saat startup, cleanup saat shutdown.
    """
    # Startup
    logger.info("FastAPI startup — loading models...")
    try:
        load_all_models()
    except FileNotFoundError as e:
        logger.error(
            f"\n{'='*60}\n"
            f"ERROR: Model file tidak ditemukan!\n{e}\n"
            f"\nPastikan file berikut ada di backend/models/:\n"
            f"  - ann_best.pt\n"
            f"  - pca_deep.pkl\n"
            f"  - scaler_deep.pkl\n"
            f"  - scaler_hc.pkl\n"
            f"  - optimal_thresholds.json\n"
            f"\nLihat ARSITEKTUR_WEB_XRAY.md untuk instruksi lengkap.\n"
            f"{'='*60}"
        )
        # Tetap jalankan server agar /health endpoint bisa diakses
        # dan memberikan pesan error yang jelas
    except Exception as e:
        logger.error(f"Model loading gagal: {e}", exc_info=True)

    yield

    # Shutdown
    logger.info("FastAPI shutdown — cleanup...")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ── FastAPI App ───────────────────────────────────────────────

app = FastAPI(
    title       = "Hybrid CNN-ANN Chest X-Ray Classifier",
    description = (
        "API untuk analisis citra X-ray dada menggunakan model hybrid "
        "CNN (DenseNet121-XRV) + Handcrafted Features (GLCM/LBP/DWT) + ANN. "
        "Dataset: NIH ChestX-ray14 (8 label penyakit)."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

# CORS — izinkan frontend Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ALLOWED_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["GET", "POST"],
    allow_headers     = ["*"],
)

# Register routes
app.include_router(router)


# ── Root endpoint ─────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name"   : "Hybrid CNN-ANN Chest X-Ray Classifier API",
        "version": "1.0.0",
        "docs"   : "/docs",
        "health" : "/api/health",
        "analyze": "/api/analyze (POST)",
    }


# ── Entrypoint ────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = False,   # Jangan pakai reload di production — model reload setiap save
        workers = 1,       # 1 worker agar model dimuat sekali (CPU, tidak ada GPU)
        log_level = "info",
    )
