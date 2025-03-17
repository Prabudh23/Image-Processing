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

def apply_gaussian_filter(image, ksize):  
    img_array = np.array(image)  
    gaussian_filtered = cv2.GaussianBlur(img_array, (ksize, ksize), 0)  
    return Image.fromarray(gaussian_filtered)  

def apply_median_filter(image, ksize):  
    img_array = np.array(image)  
    median_filtered = cv2.medianBlur(img_array, ksize)  
    return Image.fromarray(median_filtered)  

def apply_bilateral_filter(image, d, sigmaColor, sigmaSpace):  
    img_array = np.array(image)  
    bilateral_filtered = cv2.bilateralFilter(img_array, d, sigmaColor, sigmaSpace)  
    return Image.fromarray(bilateral_filtered)  

def convert_color_space(image, color_space):  
    img_array = np.array(image)  
    if color_space == 'HSV':  
        converted_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)  
    elif color_space == 'LAB':  
        converted_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)  
    return Image.fromarray(converted_image)  

def apply_canny_edge(image, threshold1, threshold2):  
    img_array = np.array(image)  
    gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)  
    edges = cv2.Canny(gray_image, threshold1, threshold2)  
    return Image.fromarray(edges)  

st.title("Advanced Image Processing App")  

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

        gaussian_ksize = st.slider("Gaussian Filter Size", 3, 15, 5, step=2)  
        if st.button("Apply Gaussian Filter"):  
            st.session_state.processed_images['gaussian'] = apply_gaussian_filter(image, gaussian_ksize)  

        median_ksize = st.slider("Median Filter Size", 3, 15, 5, step=2)  
        if st.button("Apply Median Filter"):  
            st.session_state.processed_images['median'] = apply_median_filter(image, median_ksize)  

    with col2:  
        scale_factor = st.slider("Scale Factor", 0.1, 3.0, 1.0)  
        if st.button("Scale Image"):  
            st.session_state.processed_images['scaled'] = scale_image(image, scale_factor)  

        laplacian_ksize = st.slider("Laplacian Kernel Size", 1, 7, 1, step=2)  
        if st.button("Apply Laplacian Mask"):  
            st.session_state.processed_images['laplacian'] = apply_laplacian(image, laplacian_ksize)  

        color_space = st.selectbox("Select Color Space", ["RGB", "HSV", "LAB"])  
        if st.button("Convert Color Space"):  
            st.session_state.processed_images['color_space_converted'] = convert_color_space(image, color_space)  

        canny_threshold1 = st.slider("Canny Threshold 1", 0, 255, 100)  
        canny_threshold2 = st.slider("Canny Threshold 2", 0, 255, 200)  
        if st.button("Apply Canny Edge Detection"):  
            st.session_state.processed_images['edges'] = apply_canny_edge(image, canny_threshold1, canny_threshold2)  

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
        if 'gaussian' in st.session_state.processed_images:  
            st.image(st.session_state.processed_images['gaussian'], caption="Gaussian Filtered Image", use_container_width=True)  
            st.subheader("Gaussian Filtered Image Histogram")  
            plot_histogram(st.session_state.processed_images['gaussian'])  
        if 'median' in st.session_state.processed_images:  
            st.image(st.session_state.processed_images['median'], caption="Median Filtered Image", use_container_width=True)  
            st.subheader("Median Filtered Image Histogram")  
            plot_histogram(st.session_state.processed_images['median'])  

    with col2:  
        if 'scaled' in st.session_state.processed_images:  
            st.image(st.session_state.processed_images['scaled'], caption="Scaled Image", use_container_width=True)  
            st.subheader("Scaled Image Histogram")  
            plot_histogram(st.session_state.processed_images['scaled'])  
        if 'laplacian' in st.session_state.processed_images:  
            st.image(st.session_state.processed_images['laplacian'], caption="Laplacian Image", use_container_width=True)  
            st.subheader("Laplacian Image Histogram")  
            plot_histogram(st.session_state.processed_images['laplacian'])  
        if 'color_space_converted' in st.session_state.processed_images:  
            st.image(st.session_state.processed_images['color_space_converted'], caption=f"Color Space Converted Image ({color_space})", use_container_width=True)  
        if 'edges' in st.session_state.processed_images:  
            st.image(st.session_state.processed_images['edges'], caption="Edges Detected", use_container_width=True)  
