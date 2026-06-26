import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load model yang sudah di-download dari Colab kemarin
@st.cache_resource
def load_model():
    return joblib.load('best_model_churn.pkl')

model = load_model()

st.title("🔮 Customer Churn Prediction Dashboard")
st.write("Aplikasi ini memprediksi apakah pelanggan akan berhenti menggunakan layanan (Churn) atau bertahan.")

st.divider()

# 2. Form Input Fitur Pelanggan (Sesuaikan dengan beberapa fitur penting dari EDA kamu)
st.subheader("📝 Masukkan Data Aktivitas Pelanggan")

age = st.number_input("Usia Pelanggan (Age)", min_value=10, max_value=100, value=30)
total_visits = st.number_input("Total Kunjungan (Total Visits)", min_value=0, max_value=500, value=10)
avg_session_time = st.number_input("Rata-rata Waktu Sesi (Menit)", min_value=0.0, max_value=1000.0, value=15.5)
satisfaction_score = st.slider("Skor Kepuasan (1-5)", min_value=1.0, max_value=5.0, value=4.0)

# Tambahkan fitur lain yang dibutuhkan oleh model hasil preprocessing kamu di sini

# 3. Tombol Prediksi
if st.button("Jalankan Prediksi Churn"):
    # Susun data input menjadi DataFrame / Array sesuai format saat training
    # Catatan: Pastikan jumlah kolom input sama persis dengan X_train_p saat training!
    # Contoh sederhana 4 fitur (sesuaikan dengan arsitektur model finalmu):
    input_data = np.array([[age, total_visits, avg_session_time, satisfaction_score]]) 
    
    # Lakukan prediksi
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]
    
    st.divider()
    if prediction == 1:
        st.error(f"🚨 Hasil: **Pelanggan Berpotensi CHURN** (Probabilitas: {prediction_proba:.2%})")
    else:
        st.success(f"✅ Hasil: **Pelanggan Tetap BERTAHAN (Retain)** (Probabilitas Keamanan: {(1 - prediction_proba):.2%})")
