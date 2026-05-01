# app_config.py
# ============================================================
# Konstanta global untuk seluruh pipeline backend
# ============================================================

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# ── Direktori ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# ── Target Labels ─────────────────────────────────────────────
TARGET_LABELS = [
    "Atelectasis",
    "Effusion",
    "Fibrosis",
    "Infiltration",
    "Consolidation",
    "Mass",
    "Nodule",
    "No_Finding",
]
N_LABELS = len(TARGET_LABELS)

# ── Image Size ────────────────────────────────────────────────
IMG_SIZE_HC = 224  # untuk handcraft feature & deep feature (DenseNet input)
IMG_SIZE_SEG = 512  # untuk segmentasi PSPNet

# ── Device ────────────────────────────────────────────────────
# Tidak ada GPU — paksa CPU
DEVICE = "cpu"

# ── Segmentasi ────────────────────────────────────────────────
MASK_THRESHOLD = 0.5  # threshold prob → binary mask
MORPH_KERNEL_SIZE = 7  # kernel morfologi cleanup

# ── Model File Paths ──────────────────────────────────────────
ANN_MODEL_PATH = MODELS_DIR / "ann_best.pt"
PCA_DEEP_PATH = MODELS_DIR / "pca_deep.pkl"
SCALER_DEEP_PATH = MODELS_DIR / "scaler_deep.pkl"
SCALER_HC_PATH = MODELS_DIR / "scaler_hc.pkl"
OPTIMAL_THRESHOLDS_PATH = MODELS_DIR / "optimal_thresholds.json"

# ── LLM ───────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_MAX_TOKENS = 20000

# ── GradCAM ───────────────────────────────────────────────────
# Layer target untuk GradCAM pada DenseNet121-XRV
# denseblock4 adalah blok terakhir sebelum classifier — paling semantik
GRADCAM_TARGET_LAYER = "densenet121.features.denseblock4"

# ── DenseNet Feature Dim ──────────────────────────────────────
DEEP_FEATURE_DIM = 1024

# ── Handcrafted Feature Dim ───────────────────────────────────
# GLCM(13) + LBP(10) + DWT(12) = 35
HC_FEATURE_DIM = 35

# ── Performance ───────────────────────────────────────────────
# Optimasi inferensi CPU
TORCH_NUM_THREADS = 4

# ── CORS ──────────────────────────────────────────────────────
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
