import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:5001")

FEATURES = ["radius", "texture", "perimeter", "area", "smoothness", "compactness",
            "concavity", "concave points", "symmetry", "fractal dimension"]

SAMPLES = {
    "Malignant sample": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
                         1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                         25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189],
    "Benign sample": [13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766,
                      0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023,
                      15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259],
}

st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🩺", layout="wide")
st.title("Breast Cancer Classifier")
st.caption("Streamlit UI container calling a Flask API container, which serves a model trained in a separate trainer container.")

# Check the API and show model metrics
try:
    info = requests.get(f"{API_URL}/model-info", timeout=5).json()
except requests.RequestException:
    st.error(f"Cannot reach the API at {API_URL}. Is the api container running?")
    st.stop()

st.sidebar.success("API connected")
st.sidebar.write(f"API URL: `{API_URL}`")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Model", "Gradient Boosting")
c2.metric("Accuracy", info["accuracy"])
c3.metric("F1", info["f1"])
c4.metric("ROC AUC", info["roc_auc"])

# Inputs: pick a sample to prefill, then edit any value
st.sidebar.header("Input")
sample = st.sidebar.radio("Start from", list(SAMPLES.keys()))
values = SAMPLES[sample]

st.subheader("Tumor features")
features = []
for group, tab in enumerate(st.tabs(["Mean", "Standard error", "Worst"])):
    with tab:
        cols = st.columns(5)
        for i, name in enumerate(FEATURES):
            idx = group * 10 + i
            features.append(
                cols[i % 5].number_input(
                    name.title(), value=float(values[idx]), format="%.5f", key=f"{sample}-{idx}"
                )
            )

# Predict through the API
if st.button("Predict", type="primary"):
    res = requests.post(f"{API_URL}/predict", json={"features": features}, timeout=10)
    if res.ok:
        result = res.json()["results"][0]
        label = result["prediction"].upper()
        if result["prediction"] == "malignant":
            st.error(f"Prediction: {label}")
        else:
            st.success(f"Prediction: {label}")
        st.progress(result["probability"], text=f"Confidence: {result['probability'] * 100:.1f}%")
    else:
        st.error(res.json().get("error", "Prediction failed"))

st.caption("For learning purposes only. Not a medical tool.")