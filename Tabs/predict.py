import streamlit as st
from web_function import predict

def app(df, x, y):

    st.title("Halaman Prediksi")

    # List nama fitur sesuai dataset
    feature_names = [
        "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu", "sc", "sod", "pot",
        "hemo", "pcv", "wc", "rc", "htn", "dm", "cad", "appet", "pe", "ane"
    ]

    # Membuat input form untuk setiap fitur
    st.subheader("Masukkan Nilai Fitur:")
    cols = st.columns(3)
    features = []

    for idx, feature in enumerate(feature_names):
        col = cols[idx % 3]  # Mengatur input ke dalam 3 kolom
        value = col.text_input(f"Input Nilai {feature}", value="0")
        try:
            # Konversi ke float, default 0 jika gagal
            value = float(value.replace(",", "."))  # Mengganti koma dengan titik jika ada
        except ValueError:
            value = 0.0
            st.warning(f"Nilai untuk {feature} tidak valid, menggunakan default 0.")
        features.append(value)

    # Tombol prediksi
    if st.button("Prediksi"):
        try:
            prediction, score = predict(x, y, features)
            st.info("Prediksi Sukses...")

            if prediction[0] == 1:
                st.warning("Orang Tersebut Rentan Terhadap Penyakit Batu Ginjal.")
            else:
                st.success("Orang Tersebut Relatif Aman dari Penyakit Batu Ginjal.")

            st.write("Model yang Digunakan Memiliki Tingkat Akurasi =", round(score * 100, 2), "%")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
