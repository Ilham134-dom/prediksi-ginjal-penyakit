import streamlit as st
from web_function import predict
import numpy as np
import joblib


model = joblib.load('decision_tree_model.pkl')

def app():

    st.title("Halaman Prediksi")

    # List nama fitur sesuai dataset
    feature_names = [
        "Blood Pressure(mm/Hg)", "Spesific gravity(1.005,1.010,1.015,1.020,1.025)",  "Blood Urea(mgs/dl)", "Serum Creatine(mgs/dl)", "Sodium(mEq/dL)",
        "Hemogoblin(gms)", "Packed Cell Volume(%)", "White Blood Cell Count(cells/cmm)", "Red Blood Cell Count(millions/cmm)", "Hypertension(Yes/No)", "Diabetes Mellitus(Yes/No)", "Coronary Artery Disease(Yes/No)", "Appetite(Good/Poor)", "Pedal Edema(Yes/No)", "Anemia(Yes/No)"
    ]

    # Membuat input form untuk setiap fitur
    st.subheader("Masukkan Nilai Fitur:")
    cols = st.columns(3)
    features = []

    for idx, feature in enumerate(feature_names):
        col = cols[idx % 3]  # Mengatur input ke dalam 3 kolom
        try:
            if feature not in ["Hypertension(Yes/No)", "Diabetes Mellitus(Yes/No)", "Coronary Artery Disease(Yes/No)", "Appetite(Good/Poor)", "Pedal Edema(Yes/No)", "Anemia(Yes/No)"]:
                value = col.text_input(f"Input Nilai {feature}", value="0")
                # Konversi ke float, default 0 jika gagal
                value = float(value.replace(",", "."))  # Mengganti koma dengan titik jika ada
                if feature == "Blood Pressure(mm/Hg)":
                    if value < 40 or value > 300:
                        raise ValueError("Nilai harus berada di antara 40-300 mm/Hg.")
                if feature == "Spesific gravity(1.005,1.010,1.015,1.020,1.025)":
                    if value not in [1.005,1.010,1.015,1.020,1.025]:
                        raise ValueError("Nilai harus berupa 1.005, 1.010, 1.015, 1.020, atau 1.025.")
                if feature == "Blood Urea(mgs/dl)":
                    if value < 1 or value > 150:
                        raise ValueError("Nilai harus berada di antara 1-150 mgs/dl.")
                if feature == "Serum Creatine(mgs/dl)":
                    if value < 0.2 or value > 15:
                        raise ValueError("Nilai harus berada di antara 0.2-15 mgs/dl.")
                if feature == "Sodium(mEq/dL)":
                    if value < 120 or value > 160:
                        raise ValueError("Nilai harus berada di antara 120-160 mEq/dl.")
                # Hemoglobin(g/dL)
                if feature == "Hemoglobin(g/dL)":
                    if value < 5.0 or value > 25.0:
                        raise ValueError("Nilai harus berada di antara 5.0-25.0 g/dL.")
                # Packed Cell Volume(PCV, %)
                if feature == "Packed Cell Volume(%)":
                    if value < 10.0 or value > 65.0:
                        raise ValueError("Nilai harus berada di antara 10.0-65.0%.")
                    value /= 100
                # White Blood Cell Count (WBC, cells/cumm)
                if feature == "White Blood Cell Count(cells/cumm)":
                    if value < 1000 or value > 30000:
                        raise ValueError("Nilai harus berada di antara 1000-30000 cells/cumm.")
                # Red Blood Cell Count (RBC, millions/cmm)
                if feature == "Red Blood Cell Count(millions/cmm)":
                    if value < 2.0 or value > 7.5:
                        raise ValueError("Nilai harus berada di antara 2.0-7.5 million cells/cmm.")
            else:
                if feature != "Appetite(Good/Poor)":  
                    value = col.selectbox(feature, ['Yes', 'No'])
                    if(value == 'Yes'):
                        value = 1
                    else:
                        value = 0
                else:
                    value = col.selectbox(feature, ['Good', 'Poor'])
                    if(value == 'Good'):
                        value = 0
                    else:
                        value = 1
                
            
            
        except ValueError as e:
            value = 0.0
            st.warning(f"Nilai untuk {feature} tidak valid, {e}.")
        except Exception as e:
            st.warning(f"Nilai untuk {feature} tidak valid, {e}.")
        features.append(value)

    features = np.array(features).reshape(1, -1).astype(float)
    # Tombol prediksi
    if st.button("Prediksi"):
        try:
            prediction = model.predict(features) 
            st.info("Prediksi Sukses...")

            if prediction == 0:
                st.warning("Orang Tersebut Rentan terhadap Penyakit Ginjal Kronis.")
            else:
                st.success("Orang Tersebut Relatif Aman dari Penyakit Ginjal Kronis.")

            st.write("Model yang Digunakan Memiliki Tingkat Akurasi =", round( 0.9928 * 100, 2), "%")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
