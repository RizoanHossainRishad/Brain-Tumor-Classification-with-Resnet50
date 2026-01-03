import streamlit as st
import requests
from PIL import Image

# -----------------------------
# Config
# -----------------------------
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Brain Tumour Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumour Classification")
st.subheader("By Rizoan Hossain Rishad")
st.write("Upload an MRI image to predict the tumour type. Types can be glioma, meningioma, pituitary, notumour")

# -----------------------------
# File uploader
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    if st.button("🔍 Predict By RESNET50"):
        with st.spinner("Sending image to model..."):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            result = response.json()

            st.success("Prediction completed")

            st.subheader("🧪 Result")
            st.write(f"**Class:** `{result['class_name']}`")
            st.write(f"**Confidence:** `{result['confidence'] * 100:.2f}%`")
        else:
            st.error("Failed to get prediction from API")
