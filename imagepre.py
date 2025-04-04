import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Image Transformation Tool", layout="centered")

st.title("🖼️ Image Transformation with Histogram and Filters")
st.markdown("Apply transformations like **rotation**, **shear**, **scaling**, and **Laplacian filtering** on your image.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def show_histogram(image):
    st.subheader("📊 Histogram of Original Image")
    color = ('r', 'g', 'b')
    plt.figure(figsize=(6, 3))
    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    st.pyplot(plt)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    h, w = img_np.shape[:2]
    center = (w // 2, h // 2)

    # Show original image and histogram
    st.image(image, caption="🖼️ Original Image", use_column_width=True)
    show_histogram(img_np)

    st.sidebar.header("🔧 Transformation Settings")
    rotation_angle = st.sidebar.slider("Rotate (degrees)", -180, 180, 0)
    shear_x = st.sidebar.slider("Shear X", -1.0, 1.0, 0.0, step=0.01)
    shear_y = st.sidebar.slider("Shear Y", -1.0, 1.0, 0.0, step=0.01)
    scale = st.sidebar.slider("Scale", 0.1, 3.0, 1.0, step=0.1)
    apply_laplacian = st.sidebar.checkbox("Apply Laplacian Filter", value=False)
    laplacian_size = st.sidebar.selectbox("Laplacian Kernel Size", [3, 5]) if apply_laplacian else None

    # --- Rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated_img = cv2.warpAffine(img_np, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # --- Shearing
    shear_matrix = np.array([
        [1, shear_x, 0],
        [shear_y, 1, 0]
    ], dtype=np.float32)
    sheared_img = cv2.warpAffine(img_np, shear_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # --- Scaling
    scaled_img = cv2.resize(img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # --- Laplacian
    if apply_laplacian:
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray_img, cv2.CV_64F, ksize=laplacian_size)
        lap = cv2.convertScaleAbs(lap)
        laplacian_img = cv2.cvtColor(lap, cv2.COLOR_GRAY2RGB)

    # Show each transformed image individually
    st.subheader("🔄 Rotated Image")
    st.image(rotated_img, use_column_width=True)

    st.subheader("↔️ Sheared Image")
    st.image(sheared_img, use_column_width=True)

    st.subheader("🔍 Scaled Image")
    st.image(scaled_img, use_column_width=True)

    if apply_laplacian:
        st.subheader(f"🧠 Laplacian Filtered Image (Kernel: {laplacian_size}x{laplacian_size})")
        st.image(laplacian_img, use_column_width=True)

    st.success("✅ All transformations applied successfully!")

else:
    st.info("Please upload an image to begin.")
