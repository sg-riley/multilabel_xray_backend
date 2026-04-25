# backend/models/ — Panduan Pengumpulan File Model

Folder ini berisi file model hasil training dari Google Colab.
**File-file ini TIDAK di-commit ke git** (tambahkan ke .gitignore).

---

## File yang Diperlukan

| File di sini         | Source dari Colab                                                    | Keterangan           |
|----------------------|----------------------------------------------------------------------|----------------------|
| `ann_best.pt`        | `06_training_eval/densenet121_xrv_all_aug/ann_densenet121_xrv_all_aug_best.pt` | Model ANN terbaik    |
| `pca_deep.pkl`       | `05_pca_fusion/densenet121_xrv_all_aug/pca_deep.pkl`                | PCA object           |
| `scaler_deep.pkl`    | `05_pca_fusion/densenet121_xrv_all_aug/scaler_deep.pkl`             | StandardScaler deep  |
| `scaler_hc.pkl`      | `05_pca_fusion/densenet121_xrv_all_aug/scaler_hc.pkl`               | StandardScaler HC    |
| `optimal_thresholds.json` | `06_training_eval/densenet121_xrv_all_aug/optimal_thresholds_densenet121_xrv_all_aug.json` | Threshold Youden's J |

---

## Cara Download dari Google Drive

### Option 1: Manual
1. Buka Google Drive → folder `[DEV]Chest_X_Ray_Model`
2. Download file-file di atas
3. Rename sesuai kolom "File di sini"
4. Taruh di folder `backend/models/`

### Option 2: Script (jalankan di terminal)
```bash
# Install gdown dulu
pip install gdown

# Contoh download (ganti FILE_ID dengan ID dari Drive URL):
gdown "https://drive.google.com/uc?id=FILE_ID" -O models/ann_best.pt
gdown "https://drive.google.com/uc?id=FILE_ID" -O models/pca_deep.pkl
gdown "https://drive.google.com/uc?id=FILE_ID" -O models/scaler_deep.pkl
gdown "https://drive.google.com/uc?id=FILE_ID" -O models/scaler_hc.pkl
gdown "https://drive.google.com/uc?id=FILE_ID" -O models/optimal_thresholds.json
```

---

## Catatan

- **PSPNet** dan **DenseNet121-XRV** TIDAK perlu di-download manual.
  Keduanya di-download otomatis oleh `torchxrayvision` saat pertama kali dijalankan.
  Cache disimpan di `~/.torchxrayvision/` atau `~/.cache/torch/`.

- **Verifikasi**: Setelah meletakkan semua file, jalankan:
  ```bash
  cd backend
  python -c "from pipeline.classifier import load_classifier; load_classifier(); print('OK')"
  ```

---

## Struktur Setelah Lengkap

```
models/
├── ann_best.pt
├── pca_deep.pkl
├── scaler_deep.pkl
├── scaler_hc.pkl
└── optimal_thresholds.json
```
