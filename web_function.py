#import modul yang akan digunakan
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
import joblib

model = joblib.load('decision_tree_model.pkl')


# @st.cache()
def load_data():
    # Load dataset
    df = pd.read_csv('kidney_clean.csv')

    # Pilih fitur dan label
    x = df[['bp', 'sg', 'bu', 'sc', 'sod', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']]
    y = df[['classification']]

    # Konversi tipe data jika diperlukan
    x = preprocess_input(x)

    return df, x, y

def preprocess_input(data):
    """
    Membersihkan data input:
    - Mengganti koma dengan titik jika ada
    - Mengubah tipe data string menjadi numerik
    - Mengganti nilai NaN dengan nilai default (0)
    """
    data = data.replace(',', '.', regex=True)  # Ganti koma dengan titik
    data = data.apply(pd.to_numeric, errors='coerce')  # Konversi ke numerik
    data = data.fillna(0)  # Ganti NaN dengan 0 atau nilai lain yang sesuai
    return data

# @st.cache()
def train_model(x, y):
    """
    Melatih model DecisionTreeClassifier dengan parameter yang ditentukan.
    """
    model = DecisionTreeClassifier(
        ccp_alpha=0.0, class_weight=None, criterion='entropy',
        max_depth=4, max_features=None, max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_samples_leaf=1,
        min_samples_split=2, min_weight_fraction_leaf=0.0,
        random_state=42, splitter='best'
    )

    # Pastikan data input valid
    x = preprocess_input(x)

    # Fit model
    model.fit(x, y)

    # Hitung skor akurasi
    score = model.score(x, y)

    return model, score

def predict(x, y, features):
    """
    Membuat prediksi menggunakan model yang telah dilatih.
    - Validasi fitur input sebelum prediksi.
    """
    # model, score = train_model(x, y)

    # Validasi input fitur
    features = preprocess_features(features, x.columns)

    # Prediksi
    prediction = model.predict(features)

    return prediction

def preprocess_features(features, columns):
    """
    Membersihkan dan memvalidasi data fitur untuk prediksi:
    - Mengubah data ke dalam format array
    - Memastikan semua data berupa float
    """
    # Jika fitur berupa dictionary atau pandas DataFrame, pastikan urutan kolom sesuai
    if isinstance(features, (pd.DataFrame, pd.Series)):
        features = features[columns].values

    # Konversi ke float
    features = np.array(features).reshape(1, -1).astype(float)

    return features
