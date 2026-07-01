import streamlit as st
import joblib
import numpy as np

# 1. Load model yang sudah di-download dari Colab
@st.cache_resource
def load_model():
    return joblib.load('best_model_churn.pkl')

model = load_model()

st.title("🔮 Customer Churn Prediction Dashboard")
st.write("Aplikasi ini memprediksi apakah pelanggan akan berhenti menggunakan layanan (Churn) atau bertahan.")

st.subheader("📝 Masukkan Data Aktivitas Pelangan")

# Fitur yang di-input user di web
age = st.number_input("Usia Pelanggan (Age)", min_value=1, max_value=100, value=30)
total_visits = st.number_input("Total Kunjungan (Total Visits)", min_value=0, value=10)
avg_session_time = st.number_input("Rata-rata Waktu Sesi (Menit)", min_value=0.0, value=15.5)
satisfaction_score = st.slider("Skor Kepuasan (1-5)", min_value=1, max_value=5, value=4)

if st.button("Jalankan Prediksi Churn"):
    # Buat array berisi angka 0 sebanyak 1839 kolom sesuai kebutuhan model
    input_features = np.zeros(1839)
    
    # Isi kolom numerik utama berdasarkan posisi indeks fiturnya
    input_features[0] = age               # 'age' berada di indeks 0
    input_features[2] = total_visits      # 'total_visits' berada di indeks 2
    input_features[3] = avg_session_time  # 'avg_session_time' berada di indeks 3
    input_features[13] = satisfaction_score # 'satisfaction_score' berada di indeks 13
    
    # Ubah menjadi bentuk matriks 2D untuk siap di-predict
    final_input = input_features.reshape(1, -1)
    
    # Lakukan Prediksi
    prediction = model.predict(final_input)[0]
    
    st.markdown("---")
    if prediction == 1:
        st.error("🚨 **Hasil Prediksi:** Pelanggan ini berpotensi besar untuk **CHURN** (Berhenti Berlangganan).")
    else:
        st.success("✅ **Hasil Prediksi:** Pelanggan ini diprediksi akan **RETAIN** (Tetap Bertahan).")
