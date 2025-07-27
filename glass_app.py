import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Page settings
st.set_page_config(page_title="Glass Type Classifier", layout="centered")
st.title("🔍 Glass Type Classifier App")
st.write("Predict glass types using chemical compositions and visualize the dataset.")

# Load model and data
model = joblib.load("glass_classifier.pkl")
df = pd.read_csv("glass.csv")

# --- Sidebar for Navigation ---
page = st.sidebar.selectbox("Select Page", ["🔮 Prediction", "📊 Heatmap", "📉 Confusion Matrix"])

# --- Prediction Page ---
if page == "🔮 Prediction":
    st.subheader("Input Chemical Properties")

    RI = st.number_input("Refractive Index (RI)", min_value=1.4, max_value=1.6, step=0.001)
    Na = st.number_input("Sodium (Na)", min_value=0.0)
    Mg = st.number_input("Magnesium (Mg)", min_value=0.0)
    Al = st.number_input("Aluminum (Al)", min_value=0.0)
    Si = st.number_input("Silicon (Si)", min_value=0.0)
    K = st.number_input("Potassium (K)", min_value=0.0)
    Ca = st.number_input("Calcium (Ca)", min_value=0.0)
    Ba = st.number_input("Barium (Ba)", min_value=0.0)
    Fe = st.number_input("Iron (Fe)", min_value=0.0)

    if st.button("Predict Glass Type"):
        input_data = np.array([[RI, Na, Mg, Al, Si, K, Ca, Ba, Fe]])
        prediction = model.predict(input_data)[0]
        st.success(f"🔮 Predicted Glass Type: {prediction}")

# --- Heatmap Page ---
elif page == "📊 Heatmap":
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# --- Confusion Matrix Page ---
elif page == "📉 Confusion Matrix":
    st.subheader("Model Confusion Matrix")

    # Prepare features and labels
    X = df.drop('Type', axis=1)
    y = df['Type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(ax=ax2, cmap='Blues')
    plt.title("Confusion Matrix")
    st.pyplot(fig2)