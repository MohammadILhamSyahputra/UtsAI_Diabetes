# ======================================================
# TRAIN MODEL MLP - KLASIFIKASI PENYAKIT DIABETES MELITUS
# Target: diabetes
# 0 = Tidak Diabetes
# 1 = Diabetes
# Sumber dataset: Kaggle - Diabetes Prediction Dataset (Mohammed Mustafa)
# https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset
# Jurnal rujukan: JUKTISI Vol 5 No 1, Juni 2026 (hal 401-411)
# ======================================================

import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# ======================================================
# 1. LOAD DATASET
# ======================================================
file_path = "diabetes_prediction_dataset.csv"
df = pd.read_csv(file_path)

print("=" * 60)
print("DATASET")
print("=" * 60)
print("Jumlah data :", len(df))
print("Jumlah kolom:", len(df.columns))
print()
print(df.head())
print()

# ======================================================
# 2. CEK DATA
# ======================================================
print("=" * 60)
print("INFORMASI DATA")
print("=" * 60)
print(df.info())
print()
print("Missing value:")
print(df.isnull().sum())
print()
n_awal = len(df)
n_duplikat = int(df.duplicated().sum())
kelas_awal = df["diabetes"].value_counts()
print("Jumlah data duplikat:", n_duplikat)
print()
print("Distribusi kelas diabetes:")
print(df["diabetes"].value_counts())
print()

# ======================================================
# 3. MEMBERSIHKAN DATA
# ======================================================
df = df.dropna().copy()

# Menghapus data duplikat
df = df.drop_duplicates().reset_index(drop=True)

n_bersih = len(df)
print("Jumlah data setelah cleaning:", n_bersih)
print()

# ======================================================
# 4. MEMISAHKAN FITUR DAN TARGET
# ======================================================
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# ======================================================
# 5. MENENTUKAN KOLOM NUMERIK DAN KATEGORIKAL
# ======================================================
numeric_features = [
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level"
]

categorical_features = [
    "gender",
    "smoking_history"
]

# ======================================================
# 6. PREPROCESSING
# ======================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
    ]
)

# ======================================================
# 7. MEMBAGI DATA TRAINING DAN TESTING
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("=" * 60)
print("PEMBAGIAN DATA")
print("=" * 60)
print("Data training:", X_train.shape)
print("Data testing :", X_test.shape)
print()

# ======================================================
# 8. MEMBUAT MODEL MLP
# ======================================================
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64, 32),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10
)

# ======================================================
# 9. PIPELINE
# ======================================================
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("mlp", mlp)
    ]
)

# ======================================================
# 10. TRAINING
# ======================================================
print("=" * 60)
print("TRAINING MLP")
print("=" * 60)
model.fit(X_train, y_train)
print("Training selesai.")
print("Jumlah iterasi:", model.named_steps["mlp"].n_iter_)
print()

# ======================================================
# 11. PREDIKSI DATA TEST
# ======================================================
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ======================================================
# 12. EVALUASI (Accuracy, Precision, Recall, Confusion Matrix)
# ======================================================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("=" * 60)
print("HASIL EVALUASI")
print("=" * 60)
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Tidak Diabetes", "Diabetes"]))
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print()

# ======================================================
# 13. GRAFIK CONFUSION MATRIX
# Gambar disimpan di folder static agar tampil di aplikasi web
# ======================================================
os.makedirs("static", exist_ok=True)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["Tidak Diabetes", "Diabetes"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix MLP")
plt.tight_layout()
plt.savefig("static/confusion_matrix.png", dpi=300)
plt.show()

# ======================================================
# 14. GRAFIK TRAINING LOSS
# ======================================================
plt.figure(figsize=(8, 6))
plt.plot(model.named_steps["mlp"].loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("MLP Training Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("static/training_loss.png", dpi=300)
plt.show()

# ======================================================
# 15. SIMPAN MODEL
# ======================================================
joblib.dump(model, "model_mlp.pkl")
print("\nModel disimpan ....")

# ======================================================
# 16. SIMPAN RINGKASAN HASIL (dibaca oleh aplikasi web)
# ======================================================
ringkasan = {
    "n_awal": n_awal,
    "n_duplikat": n_duplikat,
    "n_bersih": n_bersih,
    "kelas_0": int(kelas_awal.get(0, 0)),
    "kelas_1": int(kelas_awal.get(1, 0)),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "n_iter": int(model.named_steps["mlp"].n_iter_),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "roc_auc": float(roc_auc),
    "cm": cm.tolist()
}
with open("metrics.json", "w") as f:
    json.dump(ringkasan, f, indent=2)
print("Ringkasan hasil disimpan ke metrics.json")