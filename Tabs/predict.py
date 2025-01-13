import streamlit as st
from web_function import predict

def app(df, x, y):

    st.title("Halaman Prediksi")

    # List nama fitur sesuai dataset
    feature_names = [
        "Blood Presure (mm/hg)", "Spesific gravity(1.005,1.010,1.015,1.020,1.025)", "Albumin (0,1,2,3,4,5)", "Sugar(0,1,2,3,4,5)", "Red Blood Cells(0/1)", "Pus Cell(0/1)", "Pus Cell Clumps(0/1)", "Bacteria(0/1)", "Blood Glucose Random(mgs/dl)", "Blood Urea(mgs/dl)", "Serum Creatine(mgs/dl)", "Sodium(mEq/dL)", "Potassium(mEq/L)",
        "Hemogoblin(gms)", "Packed Cell Volume(%)", "White Blood Cell Count(cells/cumm)", "Red Blood Cell Count(millions/cmm)", "Hypertension(0/1)", "Diabetes Mellitus(0/1)", "Coronary Artery Disease(0/1)", "Appetite(0/1)", "Pedal Edema(0/1)", "Anemia(0/1)"
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
            prediction = predict(x, y, features)
            st.info("Prediksi Sukses...")

            if prediction[0] == 1:
                st.warning("Orang Tersebut Rentan terhadap Penyakit Ginjal Kronis.")
            else:
                st.success("Orang Tersebut Relatif Aman dari Penyakit Ginjal Kronis.")

            st.write("Model yang Digunakan Memiliki Tingkat Akurasi =", round( 0.9928 * 100, 2), "%")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
