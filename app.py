import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import joblib
import json
import os

st.set_page_config(page_title="Pneumonia Detector", layout="wide")
st.title("🩺 Pneumonia Detector in Chest X-Rays")

st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox(
    "Select prediction model",
    [
        "CNN (from scratch)",
        "MobileNetV2 + Random Forest",
        "MobileNetV2 Fine-tuned"
    ]
)

MODEL_KEYS = {
    "CNN (from scratch)":          "CNN",
    "MobileNetV2 + Random Forest": "MobileNetV2 + RF",
    "MobileNetV2 Fine-tuned":      "MobileNetV2 Fine-tuned"
}

if os.path.exists('model_metrics.json'):
    with open('model_metrics.json') as f:
        all_metrics = json.load(f)
    metric_key = MODEL_KEYS[model_choice]
    if metric_key in all_metrics:
        m = all_metrics[metric_key]
        st.sidebar.markdown("### Test Set Metrics")
        st.sidebar.metric("Accuracy", f"{m['accuracy']:.2%}")
        c1, c2 = st.sidebar.columns(2)
        c1.metric("Precision", f"{m['precision']:.2%}")
        c2.metric("Recall",    f"{m['recall']:.2%}")
        st.sidebar.metric("F1-Score", f"{m['f1']:.2%}")
        st.sidebar.caption(
            "**Recall** measures the proportion of true pneumonia cases correctly detected. "
            "In medical diagnosis it takes priority over precision."
        )


@st.cache_resource
def load_cnn():
    return load_model('modelo_neumonia.h5')

@st.cache_resource
def load_mobilenet_extractor():
    return load_model('modelo_mobilenet_features.h5')

@st.cache_resource
def load_rf():
    return joblib.load('modelo_rf.pkl')

@st.cache_resource
def load_finetuned():
    return load_model('modelo_mobilenet_finetuned.h5')


uploaded_file = st.file_uploader("Upload a chest X-ray...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert('RGB')
        img_resized = img.resize((150, 150))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if model_choice == "CNN (from scratch)":
            prob_pneumonia = float(load_cnn().predict(img_array, verbose=0)[0][0])

        elif model_choice == "MobileNetV2 + Random Forest":
            features = load_mobilenet_extractor().predict(img_array, verbose=0)
            rf = load_rf()
            if hasattr(rf, 'predict_proba'):
                prob_pneumonia = float(rf.predict_proba(features)[0][1])
            else:
                prob_pneumonia = float(rf.predict(features)[0])

        else:
            prob_pneumonia = float(load_finetuned().predict(img_array, verbose=0)[0][0])

        prob_normal = 1 - prob_pneumonia

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Analysed image", width=300)
            st.markdown(f"**Model used:** {model_choice}")
            if prob_pneumonia > 0.5:
                st.error(f"🚨 **Pneumonia detected** — probability: {prob_pneumonia:.2%}")
            else:
                st.success(f"✅ **No pneumonia** — normal probability: {prob_normal:.2%}")

        with col2:
            fig, ax = plt.subplots(figsize=(5, 3))
            colors = ['#2ecc71', '#e74c3c']
            bars = ax.barh(['Normal', 'Pneumonia'], [prob_normal, prob_pneumonia], color=colors)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            ax.bar_label(bars, fmt='%.2f', padding=4)
            ax.set_title("Prediction result")
            fig.tight_layout()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

st.markdown("---")
st.caption("**Note:** This tool is for guidance only. Always consult a medical professional to validate any diagnosis.")
