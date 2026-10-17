# ======================================================
# APLIKASI WEB FLASK - PREDIKSI DIABETES MELITUS (MLP)
# Model: model_mlp.pkl (hasil Train_model.py)
# ======================================================

import os
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)


@app.template_filter("ribuan")
def ribuan(angka):
    """Format angka dengan titik pemisah ribuan, contoh 96146 -> 96.146"""
    return f"{int(angka):,}".replace(",", ".")


# ======================================================
# 1. LOAD MODEL
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model_mlp.pkl"))

# ======================================================
# 1b. RINGKASAN HASIL TRAINING
# Dibaca dari metrics.json (dibuat oleh Train_model.py).
# Jika file belum ada, dipakai nilai bawaan di bawah.
# ======================================================
METRICS_DEFAULT = {
    "n_awal": 100000, "n_duplikat": 3854, "n_bersih": 96146,
    "kelas_0": 91500, "kelas_1": 8500,
    "n_train": 76916, "n_test": 19230, "n_iter": None,
    "accuracy": 0.9704, "precision": 0.9788, "recall": 0.6792,
    "f1": 0.8019, "roc_auc": 0.9749,
    "cm": [[17509, 25], [544, 1152]],
}


def muat_metrics():
    metrics = dict(METRICS_DEFAULT)
    path = os.path.join(BASE_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            metrics.update(json.load(f))
    return metrics


METRICS = muat_metrics()

# Hasil penelitian pada jurnal rujukan (JUKTISI Vol 5 No 1, Juni 2026)
JURNAL = {"accuracy": 0.9716, "precision": 0.9940, "recall": 0.6816,
          "f1": 0.8087, "roc_auc": 0.9750}

# Contoh 5 baris pertama dataset
SAMPEL_DATA = [
    ("Female", 80, 0, 1, "never", 25.19, 6.6, 140, 0),
    ("Female", 54, 0, 0, "No Info", 27.32, 6.6, 80, 0),
    ("Male", 28, 0, 0, "never", 27.32, 5.7, 158, 0),
    ("Female", 36, 0, 0, "current", 23.45, 5.0, 155, 0),
    ("Male", 76, 1, 1, "current", 20.14, 4.8, 155, 0),
]

# ======================================================
# 2. PILIHAN FORM
# ======================================================
GENDER_OPTIONS = [("Female", "Perempuan"), ("Male", "Laki-laki"), ("Other", "Lainnya")]
YESNO_OPTIONS = [("0", "Tidak"), ("1", "Ya")]
SMOKING_OPTIONS = [
    ("never", "Tidak pernah merokok"),
    ("former", "Mantan perokok"),
    ("current", "Perokok aktif"),
    ("ever", "Pernah merokok"),
    ("not current", "Tidak merokok saat ini"),
    ("No Info", "Tidak ada informasi"),
]

# Rentang nilai yang diterima (label, minimum, maksimum)
RANGES = {
    "age": ("Usia", 0.08, 120),
    "bmi": ("BMI", 10, 70),
    "HbA1c_level": ("Kadar HbA1c", 3, 15),
    "blood_glucose_level": ("Glukosa darah", 50, 500),
}


def baca_form(form):
    """Membaca dan memvalidasi input form. Mengembalikan (data, daftar_error)."""
    errors = []
    data = {
        "gender": form.get("gender", ""),
        "hypertension": form.get("hypertension", ""),
        "heart_disease": form.get("heart_disease", ""),
        "smoking_history": form.get("smoking_history", ""),
    }

    if data["gender"] not in [v for v, _ in GENDER_OPTIONS]:
        errors.append("Pilih jenis kelamin.")
    if data["hypertension"] not in ("0", "1"):
        errors.append("Pilih riwayat hipertensi.")
    if data["heart_disease"] not in ("0", "1"):
        errors.append("Pilih riwayat penyakit jantung.")
    if data["smoking_history"] not in [v for v, _ in SMOKING_OPTIONS]:
        errors.append("Pilih riwayat merokok.")

    for nama, (label, minimum, maksimum) in RANGES.items():
        mentah = form.get(nama, "").strip().replace(",", ".")
        try:
            nilai = float(mentah)
        except ValueError:
            errors.append(f"{label} harus berupa angka.")
            data[nama] = mentah
            continue
        if not (minimum <= nilai <= maksimum):
            errors.append(f"{label} harus antara {minimum:g} dan {maksimum:g}.")
        data[nama] = mentah

    return data, errors


# ======================================================
# 3. HALAMAN UTAMA
# ======================================================
@app.route("/", methods=["GET", "POST"])
def index():
    hasil = None
    errors = []
    data = {}

    if request.method == "POST":
        data, errors = baca_form(request.form)

        if not errors:
            # Membuat DataFrame dengan nama kolom yang sama seperti saat training
            data_input = pd.DataFrame([{
                "gender": data["gender"],
                "age": float(data["age"]),
                "hypertension": int(data["hypertension"]),
                "heart_disease": int(data["heart_disease"]),
                "smoking_history": data["smoking_history"],
                "bmi": float(data["bmi"]),
                "HbA1c_level": float(data["HbA1c_level"]),
                "blood_glucose_level": float(data["blood_glucose_level"]),
            }])

            # Prediksi
            prediksi = int(model.predict(data_input)[0])
            probabilitas = model.predict_proba(data_input)[0]

            hasil = {
                "kelas": prediksi,
                "label": "Berisiko diabetes" if prediksi == 1 else "Tidak terdeteksi diabetes",
                "p_diabetes": round(float(probabilitas[1]) * 100, 2),
                "p_normal": round(float(probabilitas[0]) * 100, 2),
            }

    static_dir = os.path.join(BASE_DIR, "static")
    return render_template(
        "index.html",
        metrics=METRICS,
        jurnal=JURNAL,
        sampel=SAMPEL_DATA,
        ada_cm=os.path.exists(os.path.join(static_dir, "confusion_matrix.png")),
        ada_loss=os.path.exists(os.path.join(static_dir, "training_loss.png")),
        hasil=hasil,
        errors=errors,
        data=data,
        gender_options=GENDER_OPTIONS,
        yesno_options=YESNO_OPTIONS,
        smoking_options=SMOKING_OPTIONS,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
