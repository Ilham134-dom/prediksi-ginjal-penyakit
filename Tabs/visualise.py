import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree
import streamlit as st
from web_function import train_model
import joblib

model = joblib.load('decision_tree_model.pkl')
def app(df, x, y):
    warnings.filterwarnings('ignore')

    st.title("Halaman Visualisasi Prediksi Penyakit Ginjal Kronis")

    # Plot Confusion Matrix
    if st.checkbox("Plot Confusion Matrix"):
        # model, score = train_model(x, y)
        y_pred = model.predict(x)
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

        plt.figure(figsize=(10, 6))
        disp.plot(cmap="viridis", values_format='d')
        plt.title("Confusion Matrix")
        st.pyplot(plt)

    # Plot Decision Tree
    if st.checkbox("Plot Decision Tree"):
        # model, score = train_model(x, y)
        dot_data = tree.export_graphviz(
            decision_tree=model, max_depth=4, out_file=None, filled=True, rounded=True,
            feature_names=x.columns, class_names=['nonckd', 'ckd']
        )
        st.graphviz_chart(dot_data)
