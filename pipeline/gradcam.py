# pipeline/gradcam.py
# ============================================================
# GradCAM untuk visualisasi fitur DenseNet121-XRV
#
# Library: pytorch-grad-cam (Jacob Gildenblat)
# pip install grad-cam
#
# Strategy:
# - Target layer: densenet121.features.denseblock4
# - GradCAM dijalankan per label yang TERDETEKSI saja
# - Karena model downstream adalah ANN (bukan DenseNet), kita perlu
#   custom target function yang mengaitkan output DenseNet ke label.
#
# Custom Target Function:
# DenseNet XRV punya classifier sendiri untuk 18 label NIH.
# Kita manfaatkan ini karena lebih semantik, walau label berbeda dari
# ANN kita. Alternatif: gunakan EigenCAM yang tidak butuh gradient label.
#
# SOLUSI: Gunakan GradCAM dengan target = channel activation maksimum
# (EigenCAM) karena ANN kita tidak directly terhubung ke DenseNet output.
# EigenCAM lebih stable dan tidak butuh classifier head yang sesuai.
# ============================================================

import cv2
import numpy as np
import torch
import logging
from typing import Dict, Optional

# pytorch-grad-cam
try:
    from pytorch_grad_cam import GradCAM, EigenCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning(
        "pytorch-grad-cam tidak terinstall. GradCAM tidak tersedia.\n"
        "Install dengan: pip install grad-cam"
    )

from app_config import TARGET_LABELS, DEVICE, IMG_SIZE_HC
from pipeline.feature_deep import get_densenet_model, get_target_layer
from pipeline.preprocessor import image_to_base64

logger = logging.getLogger(__name__)


def _get_cam_input(img_enhanced: np.ndarray) -> tuple:
    """
    Preprocess gambar untuk input GradCAM.

    Returns:
        input_tensor : torch.Tensor (1, 1, 224, 224)
        rgb_img      : numpy (224, 224, 3) float32 [0, 1] — untuk overlay
    """
    import torchvision
    import torchxrayvision as xrv

    # Preprocess identik dengan feature_deep._preprocess_for_densenet
    img_norm = xrv.datasets.normalize(img_enhanced, 255)
    img_norm = img_norm[None, ...]  # (1, H, W)

    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(IMG_SIZE_HC),
    ])
    img_224 = transform(img_norm)  # (1, 224, 224)

    input_tensor = torch.from_numpy(img_224).unsqueeze(0).float().to(DEVICE)

    # Buat RGB image untuk overlay (normalize ke [0, 1])
    img_disp = img_224[0]  # (224, 224), range [-1024, 1024]
    img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min() + 1e-8)
    img_disp = img_disp.astype(np.float32)
    rgb_img  = np.stack([img_disp, img_disp, img_disp], axis=-1)  # (224, 224, 3)

    return input_tensor, rgb_img


class _DenseNetWithSingleOutput(torch.nn.Module):
    """
    Wrapper DenseNet untuk GradCAM agar output berupa scalar per label.
    
    DenseNet XRV punya 18 output (label NIH). Kita bungkus agar bisa
    digunakan dengan ClassifierOutputTarget dari pytorch-grad-cam.
    
    Untuk label yang tidak ada di DenseNet XRV (mis. No_Finding),
    kita gunakan channel activation terkuat (EigenCAM-style).
    """

    # Mapping dari TARGET_LABELS kita ke indeks DenseNet XRV (18 label NIH)
    # DenseNet XRV label order (dari xrv.models.DenseNet.pathologies)
    XRV_LABEL_MAP = {
        "Atelectasis"   : "Atelectasis",
        "Effusion"      : "Effusion",
        "Fibrosis"      : "Fibrosis",
        "Infiltration"  : "Infiltration",
        "Consolidation" : "Consolidation",
        "Mass"          : "Mass",
        "Nodule"        : "Nodule",
        "No_Finding"    : None,  # Tidak ada di DenseNet XRV
    }

    def __init__(self, densenet_model, label: str):
        super().__init__()
        self.model = densenet_model
        self.label = label

        # Cari indeks label di model XRV
        xrv_label = self.XRV_LABEL_MAP.get(label)
        if xrv_label and hasattr(densenet_model, "pathologies"):
            try:
                pathologies = [p.lower() for p in densenet_model.pathologies]
                self.label_idx = pathologies.index(xrv_label.lower())
            except ValueError:
                self.label_idx = None
        else:
            self.label_idx = None

    def forward(self, x):
        # model.predict() return sigmoid probabilities untuk semua pathologies
        # model.features() return post-GAP feature vector
        if self.label_idx is not None and hasattr(self.model, 'predict'):
            probs = self.model.predict(x)          # (B, 18)
            return probs[:, self.label_idx:self.label_idx+1]
        else:
            # Fallback: pakai post-GAP feature mean (EigenCAM tidak butuh ini)
            feat = self.model.features(x)          # (B, 1024)
            return feat.mean(dim=1, keepdim=True)


def generate_gradcam(
    img_enhanced: np.ndarray,
    predictions: Dict[str, bool],
    probabilities: Optional[Dict[str, float]] = None,
) -> Dict[str, str]:
    """
    Generate GradCAM overlay untuk setiap label yang TERDETEKSI.

    Args:
        img_enhanced  : numpy (H, W) uint8, gambar setelah enhancement
        predictions   : dict {label: bool} — output dari classifier.classify()
        probabilities : dict {label: float} — opsional, untuk logging

    Returns:
        gradcam_images: dict {label: "data:image/png;base64,..."} 
                        hanya untuk label yang predictions[label] == True
    """
    if not GRADCAM_AVAILABLE:
        logger.warning("GradCAM tidak tersedia — mengembalikan dict kosong")
        return {}

    detected_labels = [lbl for lbl, pred in predictions.items() if pred]

    if not detected_labels:
        logger.info("Tidak ada label terdeteksi, GradCAM dilewati")
        return {}

    logger.info(f"Generating GradCAM untuk: {detected_labels}")

    gradcam_results = {}
    input_tensor, rgb_img = _get_cam_input(img_enhanced)
    base_model = get_densenet_model()
    target_layer = get_target_layer()

    for label in detected_labels:
        try:
            # Buat wrapper model untuk label ini
            wrapped_model = _DenseNetWithSingleOutput(base_model, label)
            wrapped_model.eval()

            # Gunakan EigenCAM (tidak butuh backward, lebih stable di CPU)
            # EigenCAM menggunakan PCA dari feature maps — cocok karena
            # kita punya pre-GAP features (1024, 7, 7) yang informatif
            cam = EigenCAM(
                model=wrapped_model,
                target_layers=[target_layer],
                use_cuda=(DEVICE == "cuda"),
            )

            # Generate heatmap (224, 224) float [0, 1]
            # targets=None → ambil channel terpenting (EigenCAM behavior)
            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=None,
            )
            grayscale_cam = grayscale_cam[0]  # (224, 224)

            # Overlay pada gambar original
            cam_image = show_cam_on_image(
                rgb_img,
                grayscale_cam,
                use_rgb=True,
                colormap=cv2.COLORMAP_JET,
                image_weight=0.5,
            )
            # cam_image: numpy (224, 224, 3) uint8

            # Konversi ke base64
            cam_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            gradcam_results[label] = image_to_base64(cam_bgr)

            if probabilities:
                logger.debug(
                    f"GradCAM {label}: prob={probabilities.get(label, '?'):.3f}"
                )

        except Exception as e:
            logger.error(f"GradCAM gagal untuk label '{label}': {e}", exc_info=True)
            # Jangan lempar exception — skip label ini, lanjutkan yang lain

    logger.info(f"GradCAM selesai: {len(gradcam_results)}/{len(detected_labels)} berhasil")
    return gradcam_results
