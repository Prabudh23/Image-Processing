import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Image Transformation Tool", layout="centered")

st.title("🖼️ Image Transformation with Filters")
st.markdown("Apply **rotation**, **shear**, **scaling**, and **Laplacian filter** to your uploaded image.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    st.image(image, caption="Original Image", use_column_width=True)

    st.sidebar.header("🔧 Transformation Settings")
    rotation_angle = st.sidebar.slider("Rotate (degrees)", -180, 180, 0)
    shear_x = st.sidebar.slider("Shear X", -1.0, 1.0, 0.0, step=0.01)
    shear_y = st.sidebar.slider("Shear Y", -1.0, 1.0, 0.0, step=0.01)
    scale = st.sidebar.slider("Scale", 0.1, 3.0, 1.0, step=0.1)
    apply_laplacian = st.sidebar.checkbox("Apply Laplacian Filter", value=False)

    # Get image dimensions
    h, w = img_np.shape[:2]
    center = (w // 2, h // 2)

    # Rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, scale)

    # Add shear transformation
    shear_matrix = np.array([
        [1, shear_x, 0],
        [shear_y, 1, 0]
    ], dtype=np.float32)

    # Combine rotation and shear
    transform_matrix = shear_matrix @ np.vstack([rotation_matrix, [0, 0, 1]])  # Make it 3x3 for matrix multiplication
    transform_matrix = transform_matrix[:2, :]  # Back to 2x3 for cv2.warpAffine

    transformed_image = cv2.warpAffine(img_np, transform_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    if apply_laplacian:
        gray = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian)
        processed_image = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB)
    else:
        processed_image = transformed_image

    st.markdown("### 🖌️ Transformed Image")
    st.image(processed_image, use_column_width=True)

    st.success("✅ Transformation applied successfully!")
else:
    st.info("Please upload an image to get started.")
