# Hybrid CNN-ANN Chest X-Ray Classifier — Backend

## Deskripsi

Backend FastAPI untuk klasifikasi multi-label citra X-ray dada.

**Model:** Hybrid CNN (DenseNet121-XRV pretrained) + Handcrafted Features (GLCM/LBP/DWT) + ANN  
**Dataset:** NIH ChestX-ray14 (8 label penyakit)  
**Device:** CPU only (no GPU required)

---

## Struktur Backend

```
backend/
├── main.py                        # FastAPI entry point + model loading
├── config.py                      # Konstanta global
├── requirements.txt
├── test_pipeline.py               # Test script end-to-end
├── .env.example                   # Template environment variables
│
├── api/
│   ├── __init__.py
│   └── routes.py                  # /api/analyze, /api/health, /api/labels
│
├── pipeline/
│   ├── __init__.py                # Orchestrator: run_full_pipeline()
│   ├── preprocessor.py            # enhance_xray(), load_image_from_bytes()
│   ├── segmentation.py            # PSPNet → ROI mask
│   ├── feature_handcraft.py       # GLCM(13) + LBP(10) + DWT(12) = 35 fitur
│   ├── feature_deep.py            # DenseNet121-XRV, dual output (pre/post GAP)
│   ├── fusion.py                  # PCA transform + concatenate
│   ├── classifier.py              # ANNClassifier, optimal thresholds
│   ├── gradcam.py                 # EigenCAM via pytorch-grad-cam
│   └── reporter.py                # LLM report via Groq API (Llama 3)
│
└── models/
    ├── README.md                  # Instruksi pengumpulan file model
    ├── ann_best.pt                # ← dari Colab training
    ├── pca_deep.pkl               # ← dari Colab 05_pca_fusion
    ├── scaler_deep.pkl            # ← dari Colab 05_pca_fusion
    ├── scaler_hc.pkl              # ← dari Colab 05_pca_fusion
    └── optimal_thresholds.json   # ← dari Colab 06_training_eval
```

---

## Setup & Instalasi

### 1. Buat Virtual Environment

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Catatan:** Torch + torchvision cukup besar (~2 GB). Gunakan pip dengan cache jika perlu.

### 3. Copy File Model dari Google Drive

Lihat `models/README.md` untuk instruksi lengkap. Intinya:

```
Google Drive                                       → backend/models/
─────────────────────────────────────────────────────────────────
06_training_eval/.../ann_densenet121_xrv_all_aug_best.pt → ann_best.pt
05_pca_fusion/densenet121_xrv_all_aug/pca_deep.pkl       → pca_deep.pkl
05_pca_fusion/densenet121_xrv_all_aug/scaler_deep.pkl    → scaler_deep.pkl
05_pca_fusion/densenet121_xrv_all_aug/scaler_hc.pkl      → scaler_hc.pkl
06_training_eval/.../optimal_thresholds_*.json           → optimal_thresholds.json
```

**PSPNet dan DenseNet** tidak perlu di-download manual — otomatis oleh `torchxrayvision`.

### 4. Set Environment Variables

```bash
cp .env.example .env
# Edit .env, isi GROQ_API_KEY
```

> Jika tidak ada API key, backend tetap berjalan tapi laporan LLM akan menggunakan template statis.

### 5. Test Pipeline (Opsional tapi Disarankan)

```bash
python test_pipeline.py
```

Output yang diharapkan:
```
✅ SEMUA TEST PASSED — Backend siap dijalankan!
```

### 6. Jalankan Server

```bash
python main.py
# atau
uvicorn main:app --host 0.0.0.0 --port 8000
```

Server akan tersedia di: `http://localhost:8000`

---

## API Endpoints

### `GET /` — Info
Informasi dasar API.

### `GET /api/health` — Health Check
```json
{
  "status": "ready",
  "message": "Semua model siap",
  "models": {
    "pspnet_segmentation": true,
    "densenet121_xrv": true,
    "ann_classifier": true,
    "pca_scaler": true
  }
}
```

### `POST /api/analyze` — Analisis X-Ray
**Request:** `multipart/form-data` dengan field `file` (PNG/JPG, max 10MB)

**Response:**
```json
{
  "predictions": {"Atelectasis": true, "Effusion": false, ...},
  "probabilities": {"Atelectasis": 0.734, "Effusion": 0.089, ...},
  "gradcam_images": {"Atelectasis": "data:image/png;base64,..."},
  "original_image": "data:image/png;base64,...",
  "roi_image": "data:image/png;base64,...",
  "report": "Berdasarkan analisis model AI...",
  "processing_time_ms": 5234,
  "pipeline_steps": {"preprocess_ms": 50, "segmentation_ms": 2000, ...},
  "coverage_pct": 32.5
}
```

### `GET /api/labels` — Daftar Label
```json
{
  "labels": ["Atelectasis", "Effusion", "Fibrosis", ...]
}
```

### `GET /docs` — Swagger UI
Dokumentasi interaktif FastAPI.

---

## Estimasi Performa (CPU)

| Step             | Estimasi |
|------------------|----------|
| Preprocessing    | ~50 ms   |
| Segmentasi       | ~2-4 s   |
| Handcraft        | ~200 ms  |
| Deep Feature     | ~2-4 s   |
| PCA + Fusion     | ~10 ms   |
| ANN              | <5 ms    |
| GradCAM          | ~1-2 s   |
| LLM Report       | ~2-4 s   |
| **Total**        | **~8-15 s** |

---

## Troubleshooting

### Error: File model tidak ditemukan
```
FileNotFoundError: File model tidak ditemukan: backend/models/ann_best.pt
```
→ Letakkan file model sesuai instruksi di `models/README.md`

### Error: Dimensi mismatch
```
ValueError: Input dimension mismatch: ANN=150 != Fusion=140
```
→ Pastikan `ann_best.pt` dan file `.pkl` berasal dari **training yang sama** (run yang sama di Colab)

### Error: torchxrayvision download gagal
→ Cek koneksi internet. Model DenseNet/PSPNet di-download otomatis (~80 MB total)

### GradCAM tidak muncul
→ Install `grad-cam`: `pip install grad-cam`  
→ Atau install via requirements: `pip install -r requirements.txt`

### LLM Report tidak ada / template
→ Set `GROQ_API_KEY` di `.env`

---

## Target Labels

```
Atelectasis, Effusion, Fibrosis, Infiltration,
Consolidation, Mass, Nodule, No_Finding
```
