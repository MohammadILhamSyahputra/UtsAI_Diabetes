# ======================================================
# DOCKER IMAGE - APLIKASI WEB FLASK PREDIKSI DIABETES
# ======================================================
FROM python:3.11-slim

# Jangan buat file .pyc, tampilkan log langsung
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependensi dulu (layer ini di-cache jika requirements.txt tidak berubah)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin hanya file yang dibutuhkan saat runtime
COPY app.py model_mlp.pkl metrics.json ./
COPY static/ static/
COPY templates/ templates/

EXPOSE 5000

# Jalankan dengan gunicorn (sama seperti Procfile)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--access-logfile", "-", "app:app"]
