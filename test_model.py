# ======================================================
# TEST MODEL MLP - PREDIKSI DIABETES DATA BARU
# ======================================================

import pandas as pd
import joblib

# ======================================================
# 1. LOAD MODEL
# ======================================================
model = joblib.load("model_mlp.pkl")

print("=" * 60)
print("PREDIKSI DIABETES MELITUS MENGGUNAKAN MLP")
print("=" * 60)
print()

# ======================================================
# 2. INPUT DATA PASIEN
# ======================================================
gender = input("Jenis kelamin (Female/Male/Other)      : ").capitalize()
age = float(input("Usia pasien                           : "))
hypertension = int(input("Hipertensi (0 = Tidak, 1 = Ya)       : "))
heart_disease = int(input("Penyakit jantung (0 = Tidak, 1 = Ya) : "))
smoking_history = input("Riwayat merokok (never/former/current/ever/not current/No Info): ")
bmi = float(input("BMI                                  : "))
HbA1c_level = float(input("Kadar HbA1c                          : "))
blood_glucose_level = float(input("Kadar glukosa darah (mg/dL)          : "))

# ======================================================
# 3. MEMBUAT DATAFRAME
# ======================================================
data_input = pd.DataFrame([{
    "gender": gender,
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "smoking_history": smoking_history,
    "bmi": bmi,
    "HbA1c_level": HbA1c_level,
    "blood_glucose_level": blood_glucose_level
}])

# ======================================================
# 4. PREDIKSI
# ======================================================
prediction = model.predict(data_input)[0]
probability = model.predict_proba(data_input)[0]

# ======================================================
# 5. HASIL PREDIKSI
# ======================================================
print()
print("=" * 60)
print("HASIL PREDIKSI")
print("=" * 60)
print()

print("Status Diabetes :", prediction)

if prediction == 0:
    print("Keterangan      : Tidak Diabetes")
else:
    print("Keterangan      : Diabetes")

print(f"Probabilitas    : Tidak Diabetes {probability[0]*100:.2f}% | Diabetes {probability[1]*100:.2f}%")
print()
print("=" * 60)