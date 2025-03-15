import streamlit as st
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

def plot_histogram(image):
    img_array = np.array(image.convert('L'))  # Convert to grayscale
    hist_values, bins = np.histogram(img_array.flatten(), bins=256, range=[0, 256])
    fig, ax = plt.subplots()
    ax.plot(hist_values, color='black')
    ax.set_title("Histogram")
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

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

def apply_laplacian(image, ksize):
    img_array = np.array(image)
    laplacian = cv2.Laplacian(img_array, cv2.CV_64F, ksize=ksize)
    laplacian = np.uint8(np.clip(laplacian, 0, 255))
    return Image.fromarray(laplacian)

def embed_message_dct(image, message):
    img_array = np.array(image.convert('L'))
    dct = cv2.dct(np.float32(img_array))
    message_bits = ''.join(format(ord(char), '08b') for char in message)
    for i, bit in enumerate(message_bits):
        dct[0, i] = dct[0, i] + 1 if bit == '1' else dct[0, i] - 1
    stego_image = cv2.idct(dct)
    stego_image = np.clip(stego_image, 0, 255).astype(np.uint8)
    return Image.fromarray(stego_image)

def extract_message_dct(image, length):
    img_array = np.array(image.convert('L'))
    dct = cv2.dct(np.float32(img_array))
    message_bits = ''
    for i in range(length * 8):
        message_bits += '1' if dct[0, i] > 0 else '0'
    message = ''.join(chr(int(message_bits[i:i + 8], 2)) for i in range(0, len(message_bits), 8))
    return message

st.title("Image Processing App")

if 'processed_images' not in st.session_state:
    st.session_state.processed_images = {}

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_container_width=True)
    st.subheader("Original Image Histogram")
    plot_histogram(image)

    col1, col2 = st.columns(2)
    with col1:
        rotation_angle = st.slider("Rotation Angle", 0, 360, 0)
        if st.button("Rotate Image"):
            st.session_state.processed_images['rotated'] = rotate_image(image, rotation_angle)

        shear_factor = st.slider("Shear Factor", 0.0, 1.0, 0.0)
        if st.button("Shear Image"):
            st.session_state.processed_images['sheared'] = shear_image(image, shear_factor)

    with col2:
        scale_factor = st.slider("Scale Factor", 0.1, 3.0, 1.0)
        if st.button("Scale Image"):
            st.session_state.processed_images['scaled'] = scale_image(image, scale_factor)

        laplacian_ksize = st.slider("Laplacian Kernel Size", 1, 7, 1, step=2)
        if st.button("Apply Laplacian Mask"):
            st.session_state.processed_images['laplacian'] = apply_laplacian(image, laplacian_ksize)

    st.subheader("Processed Images")
    col1, col2 = st.columns(2)
    with col1:
        if 'rotated' in st.session_state.processed_images:
            st.image(st.session_state.processed_images['rotated'], caption="Rotated Image", use_container_width=True)
            st.subheader("Rotated Image Histogram")
            plot_histogram(st.session_state.processed_images['rotated'])
        if 'sheared' in st.session_state.processed_images:
            st.image(st.session_state.processed_images['sheared'], caption="Sheared Image", use_container_width=True)
            st.subheader("Sheared Image Histogram")
            plot_histogram(st.session_state.processed_images['sheared'])
    with col2:
        if 'scaled' in st.session_state.processed_images:
            st.image(st.session_state.processed_images['scaled'], caption="Scaled Image", use_container_width=True)
            st.subheader("Scaled Image Histogram")
            plot_histogram(st.session_state.processed_images['scaled'])
        if 'laplacian' in st.session_state.processed_images:
            st.image(st.session_state.processed_images['laplacian'], caption="Laplacian Image", use_container_width=True)
            st.subheader("Laplacian Image Histogram")
            plot_histogram(st.session_state.processed_images['laplacian'])

    st.subheader("Image Steganography with DCT")
    message = st.text_input("Enter message to embed:")
    if st.button("Embed Message") and message:
        stego_image = embed_message_dct(image, message)
        st.session_state.processed_images['stego'] = stego_image
        st.image(stego_image, caption="Stego Image", use_container_width=True)

    if st.button("Extract Message"):
        extracted_message = extract_message_dct(st.session_state.processed_images['stego'], len(message))
        st.write(f"Extracted Message: {extracted_message}")
