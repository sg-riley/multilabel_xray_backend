#!/usr/bin/env python3
# test_pipeline.py
# ============================================================
# Script untuk test pipeline end-to-end SEBELUM menjalankan server.
#
# Cara pakai:
#   cd backend
#   python test_pipeline.py                     # test dengan dummy image
#   python test_pipeline.py path/to/xray.png    # test dengan gambar nyata
#
# Apa yang ditest:
#   1. Model loading (PSPNet, DenseNet, ANN, PCA)
#   2. Full pipeline end-to-end
#   3. Output validation
# ============================================================

import sys
import time
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("test_pipeline")


def create_dummy_xray_bytes(size: int = 512) -> bytes:
    """
    Buat gambar X-ray dummy (grayscale, simulasi textur paru).
    Cukup untuk test pipeline tanpa gambar nyata.
    """
    import cv2

    # Simulasi textur grayscale X-ray
    img = np.random.randint(50, 200, (size, size), dtype=np.uint8)

    # Tambah gradient seperti X-ray (tepi gelap, tengah terang)
    center = size // 2
    Y, X   = np.ogrid[:size, :size]
    dist   = np.sqrt((X - center)**2 + (Y - center)**2)
    mask   = (dist < center * 0.7).astype(np.uint8) * 150
    img    = cv2.add(img, mask)

    # Encode ke PNG bytes
    success, buffer = cv2.imencode(".png", img)
    assert success, "Gagal encode dummy image"
    return buffer.tobytes()


def test_model_loading():
    """Test bahwa semua model bisa di-load dengan benar."""
    print("\n" + "="*60)
    print("TEST 1: Model Loading")
    print("="*60)

    from pipeline.segmentation  import load_segmentation_model
    from pipeline.feature_deep  import load_deep_feature_model
    from pipeline.fusion        import load_fusion_artifacts
    from pipeline.classifier    import load_classifier, get_input_dim
    from pipeline.fusion        import get_fused_dim
    import torch

    torch.set_num_threads(4)

    steps = [
        ("PSPNet",     load_segmentation_model),
        ("DenseNet",   load_deep_feature_model),
        ("PCA/Scaler", load_fusion_artifacts),
        ("ANN",        load_classifier),
    ]

    for name, fn in steps:
        t = time.time()
        try:
            fn()
            print(f"  ✓ {name} loaded in {time.time()-t:.1f}s")
        except FileNotFoundError as e:
            print(f"  ✗ {name}: FILE NOT FOUND")
            print(f"    {e}")
            print(f"    Letakkan file model di backend/models/")
            return False
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            return False

    # Dimension check
    ann_dim    = get_input_dim()
    fused_dim  = get_fused_dim()
    print(f"\n  ANN input dim  : {ann_dim}")
    print(f"  Fused dim      : {fused_dim}")
    if ann_dim != fused_dim:
        print(f"  ✗ DIMENSI MISMATCH! ANN={ann_dim} != Fusion={fused_dim}")
        return False
    print(f"  ✓ Dimension check passed")
    return True


def test_full_pipeline(img_bytes: bytes):
    """Test full pipeline dari bytes gambar."""
    print("\n" + "="*60)
    print("TEST 2: Full Pipeline")
    print("="*60)

    from pipeline import run_full_pipeline

    print(f"  Input size: {len(img_bytes)/1024:.1f} KB")

    t = time.time()
    try:
        result = run_full_pipeline(img_bytes)
        elapsed = time.time() - t
    except Exception as e:
        import traceback
        print(f"  ✗ Pipeline error: {e}")
        traceback.print_exc()
        return False

    print(f"\n  ✓ Pipeline selesai: {elapsed*1000:.0f}ms")
    print(f"\n  Step breakdown:")
    for step, ms in result.get("pipeline_steps", {}).items():
        print(f"    {step:20s}: {ms:5d}ms")

    print(f"\n  Coverage paru: {result.get('coverage_pct', '?')}%")

    print(f"\n  Predictions:")
    for label, pred in result["predictions"].items():
        prob  = result["probabilities"][label]
        icon  = "✓" if pred else "✗"
        cam   = "📷" if label in result["gradcam_images"] else "  "
        print(f"    {icon} {cam} {label:15s}: {prob:.1%}")

    print(f"\n  GradCAM: {len(result['gradcam_images'])} images")
    print(f"  Original img: {'✓' if result.get('original_image') else '✗'}")
    print(f"  ROI img     : {'✓' if result.get('roi_image') else '✗'}")
    print(f"  Report      : {len(result.get('report',''))} chars")
    print(f"\n  Report preview:")
    report = result.get("report", "")
    print(f"    {report[:200]}...")

    return True


def test_preprocessing_only():
    """Test preprocessing pipeline saja (tanpa model loading)."""
    print("\n" + "="*60)
    print("TEST 3: Preprocessing (tanpa model)")
    print("="*60)

    from pipeline.preprocessor import preprocess_for_pipeline, image_to_base64

    dummy_bytes = create_dummy_xray_bytes(256)
    orig, enhanced = preprocess_for_pipeline(dummy_bytes)

    print(f"  ✓ Original shape  : {orig.shape}")
    print(f"  ✓ Enhanced shape  : {enhanced.shape}")
    print(f"  ✓ Enhanced range  : [{enhanced.min()}, {enhanced.max()}]")

    b64 = image_to_base64(orig)
    assert b64.startswith("data:image/png;base64,"), "Base64 format salah"
    print(f"  ✓ Base64 encode   : OK ({len(b64)} chars)")
    return True


def test_handcraft_only():
    """Test handcrafted feature extraction saja."""
    print("\n" + "="*60)
    print("TEST 4: Handcrafted Features (tanpa model)")
    print("="*60)

    import cv2
    from pipeline.feature_handcraft import extract_handcrafted_features, FEATURE_NAMES

    # Dummy ROI 224x224
    roi = np.random.randint(0, 128, (224, 224), dtype=np.uint8)

    t = time.time()
    feats = extract_handcrafted_features(roi)
    elapsed = (time.time() - t) * 1000

    print(f"  ✓ Feature shape   : {feats.shape}")
    print(f"  ✓ No NaN          : {not np.isnan(feats).any()}")
    print(f"  ✓ No Inf          : {not np.isinf(feats).any()}")
    print(f"  ✓ Time            : {elapsed:.1f}ms")
    print(f"  First 5 values    : {feats[:5].tolist()}")
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("CHEST X-RAY PIPELINE TEST")
    print("="*60)

    # Test preprocessing & handcraft dulu (tidak butuh model file)
    ok3 = test_preprocessing_only()
    ok4 = test_handcraft_only()

    # Test model loading
    ok1 = test_model_loading()

    if not ok1:
        print(
            "\n⚠️  Model loading gagal.\n"
            "Letakkan semua file model di backend/models/ terlebih dahulu.\n"
            "Lihat backend/models/README.md untuk panduan.\n"
        )
        sys.exit(1)

    # Test full pipeline
    if len(sys.argv) > 1:
        # Gunakan gambar nyata dari argument
        img_path = sys.argv[1]
        print(f"\nMenggunakan gambar: {img_path}")
        with open(img_path, "rb") as f:
            img_bytes = f.read()
    else:
        # Gunakan dummy image
        print("\nMenggunakan dummy X-ray image (512x512)")
        img_bytes = create_dummy_xray_bytes(512)

    ok2 = test_full_pipeline(img_bytes)

    print("\n" + "="*60)
    if ok1 and ok2 and ok3 and ok4:
        print("✅ SEMUA TEST PASSED — Backend siap dijalankan!")
        print("   Jalankan: python main.py")
        print("   atau    : uvicorn main:app --host 0.0.0.0 --port 8000")
    else:
        print("❌ BEBERAPA TEST GAGAL — Cek error di atas")
    print("="*60)
