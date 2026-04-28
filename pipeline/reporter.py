# pipeline/reporter.py
# ============================================================
# Pembuatan laporan radiologi sederhana via Groq API
# Model: Llama 3 (cepat dan akurat untuk laporan)
# ============================================================

import logging
from typing import Dict, Optional

from app_config import GROQ_API_KEY, LLM_MODEL, LLM_MAX_TOKENS, TARGET_LABELS

logger = logging.getLogger(__name__)

# ── System Prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """Anda adalah asisten AI yang membantu menginterpretasikan hasil analisis \
model deep learning pada citra X-ray dada. 

Tugas Anda: membuat laporan singkat dan mudah dipahami berdasarkan output model AI.

Aturan penting:
- Gunakan bahasa Indonesia yang jelas dan mudah dipahami pasien umum.
- JANGAN membuat diagnosis definitif — ini adalah hasil model AI, bukan dokter.
- SELALU sertakan disclaimer bahwa hasil ini harus dikonfirmasi oleh dokter radiologi.
- Fokus pada temuan model, bukan spekulasi klinis yang tidak didukung data.
- Format laporan: 3-4 paragraf pendek (Ringkasan Temuan, Interpretasi, Rekomendasi, Disclaimer).
- Panjang laporan: maksimal 250 kata."""


def _build_user_prompt(
    probabilities: Dict[str, float],
    predictions: Dict[str, bool],
    coverage_pct: Optional[float] = None,
) -> str:
    """Bangun user prompt dari hasil prediksi model."""

    detected     = [lbl for lbl, pred in predictions.items() if pred and lbl != "No_Finding"]
    no_finding   = predictions.get("No_Finding", False)
    not_detected = [lbl for lbl, pred in predictions.items() if not pred and lbl != "No_Finding"]

    lines = ["Hasil analisis model AI hybrid CNN-ANN pada citra X-ray dada:\n"]

    if no_finding:
        lines.append("STATUS KESELURUHAN: Model tidak mendeteksi kondisi patologis signifikan.")
    elif detected:
        lines.append(f"STATUS KESELURUHAN: Model mendeteksi {len(detected)} kondisi yang perlu diperhatikan.")
    else:
        lines.append("STATUS KESELURUHAN: Model mendeteksi kemungkinan kondisi ringan.")

    lines.append("\nKONDISI YANG TERDETEKSI (probabilitas ≥ threshold optimal):")
    if detected:
        for lbl in detected:
            prob = probabilities.get(lbl, 0)
            lines.append(f"  • {lbl}: {prob:.1%}")
    else:
        lines.append("  • Tidak ada kondisi patologis yang melebihi threshold")

    lines.append("\nKONDISI TIDAK TERDETEKSI:")
    for lbl in not_detected:
        prob = probabilities.get(lbl, 0)
        lines.append(f"  • {lbl}: {prob:.1%}")

    if coverage_pct is not None:
        lines.append(f"\nINFO SEGMENTASI: Area paru terdeteksi = {coverage_pct:.1f}% dari gambar")

    lines.append(
        "\nBuatkan laporan radiologi singkat (3-4 paragraf, maks 250 kata) "
        "berdasarkan temuan di atas. Ingat: ini adalah output model AI, bukan diagnosis dokter."
    )

    return "\n".join(lines)


def generate_report(
    probabilities: Dict[str, float],
    predictions: Dict[str, bool],
    coverage_pct: Optional[float] = None,
) -> str:
    """
    Generate laporan radiologi sederhana via Claude API.

    Args:
        probabilities: dict {label: float} dari classifier.classify()
        predictions  : dict {label: bool}  dari classifier.classify()
        coverage_pct : float opsional — % area paru dari segmentasi

    Returns:
        report_text: string laporan dalam Bahasa Indonesia

    Note:
        Jika GROQ_API_KEY tidak di-set atau terjadi error,
        fungsi ini mengembalikan laporan fallback sederhana (tidak raise exception).
    """
    # Jika API key tidak ada, kembalikan laporan template sederhana
    if not GROQ_API_KEY:
        logger.warning(
            "GROQ_API_KEY tidak di-set. Menggunakan laporan template."
        )
        return _generate_fallback_report(probabilities, predictions)

    try:
        from groq import Groq

        client      = Groq(api_key=GROQ_API_KEY)
        user_prompt = _build_user_prompt(probabilities, predictions, coverage_pct)

        logger.info(f"Generating report via {LLM_MODEL}...")
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=LLM_MAX_TOKENS,
        )

        report = completion.choices[0].message.content
        logger.info(f"Report generated ({len(report)} chars)")
        return report

    except ImportError:
        logger.error("Library 'groq' tidak terinstall. pip install groq")
        return _generate_fallback_report(probabilities, predictions)

    except Exception as e:
        logger.error(f"LLM report generation gagal: {e}", exc_info=True)
        return _generate_fallback_report(probabilities, predictions)


def _generate_fallback_report(
    probabilities: Dict[str, float],
    predictions: Dict[str, bool],
) -> str:
    """
    Laporan template sederhana tanpa LLM.
    Digunakan jika API key tidak ada atau LLM error.
    """
    detected   = [lbl for lbl, pred in predictions.items() if pred and lbl != "No_Finding"]
    no_finding = predictions.get("No_Finding", False)

    lines = ["**LAPORAN ANALISIS AI — CHEST X-RAY**\n"]

    if no_finding and not detected:
        lines.append(
            "**Ringkasan Temuan:** Model AI tidak mendeteksi kondisi patologis "
            "yang signifikan pada citra X-ray dada ini. Probabilitas semua kondisi "
            "berada di bawah threshold optimal masing-masing.\n"
        )
    elif detected:
        det_str = ", ".join(detected)
        lines.append(
            f"**Ringkasan Temuan:** Model AI mendeteksi kemungkinan adanya kondisi "
            f"berikut: {det_str}. "
            f"Hasil ini berdasarkan analisis model hybrid CNN-ANN.\n"
        )

    lines.append("**Detail Probabilitas:**")
    for label, prob in probabilities.items():
        status = "✓ Terdeteksi" if predictions.get(label) else "✗ Tidak terdeteksi"
        lines.append(f"- {label}: {prob:.1%} ({status})")

    lines.append(
        "\n**Rekomendasi:** Hasil analisis ini bersifat pendukung keputusan klinis "
        "dan tidak dapat menggantikan pemeriksaan oleh dokter radiologi yang berkompeten.\n"
    )

    lines.append(
        "**Disclaimer:** Laporan ini dihasilkan oleh model kecerdasan buatan "
        "dan BUKAN merupakan diagnosis medis. Konsultasikan dengan tenaga medis "
        "profesional untuk interpretasi dan penanganan yang tepat."
    )

    return "\n".join(lines)
