import streamlit as st
from PIL import Image
import numpy as np
import cv2

def rotate_image(image, angle):
    img_array = np.array(image)
    center = (img_array.shape[1] // 2, img_array.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(img_array, M, (img_array.shape[1], img_array.shape[0]))
    return Image.fromarray(rotated_image)

def scale_image(image, scale_factor):
    width, height = image.size
    new_size = (int(width * scale_factor), int(height * scale_factor))
    scaled_image = image.resize(new_size, Image.LANCZOS)
    return scaled_image

def shear_image(image, shear_factor):
    img_array = np.array(image)
    rows, cols, _ = img_array.shape
    M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    sheared_image = cv2.warpAffine(img_array, M, (cols, rows))
    return Image.fromarray(sheared_image)

def apply_laplacian(image):
    img_array = np.array(image)
    laplacian = cv2.Laplacian(img_array, cv2.CV_64F)
    laplacian = np.uint8(np.clip(laplacian, 0, 255))
    return Image.fromarray(laplacian)

st.title("Image Processing App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)

    col1, col2 = st.columns(2)
    with col1:
        rotation_angle = st.slider("Rotation Angle", 0, 360, 0)
        if st.button("Rotate Image"):
            rotated_image = rotate_image(image, rotation_angle)
            st.image(rotated_image, caption="Rotated Image", use_column_width=True)

        shear_factor = st.slider("Shear Factor", 0.0, 1.0, 0.0)
        if st.button("Shear Image"):
            sheared_image = shear_image(image, shear_factor)
            st.image(sheared_image, caption="Sheared Image", use_column_width=True)

    with col2:
        scale_factor = st.slider("Scale Factor", 0.1, 3.0, 1.0)
        if st.button("Scale Image"):
            scaled_image = scale_image(image, scale_factor)
            st.image(scaled_image, caption="Scaled Image", use_column_width=True)

        if st.button("Apply Laplacian Mask"):
            laplacian_image = apply_laplacian(image)
            st.image(laplacian_image, caption="Laplacian Image", use_column_width=True)
