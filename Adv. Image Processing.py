import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import matplotlib.pyplot as plt

def plot_histogram(image):
    img_array = np.array(image.convert('L'))
    hist_values, _ = np.histogram(img_array.flatten(), bins=256, range=[0, 256])
    fig, ax = plt.subplots()
    ax.plot(hist_values, color='black')
    ax.set_title("Histogram")
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

def process_image(image, operation, *args):
    try:
        img_array = np.array(image)
        processed_image = None  # Initialize variable

        if operation == 'scale':
            scale_factor = args[0]
            width = int(img_array.shape[1] * scale_factor)
            height = int(img_array.shape[0] * scale_factor)
            dim = (width, height)
            
            if scale_factor < 1.0:
                interpolation = cv2.INTER_AREA
            else:
                interpolation = cv2.INTER_CUBIC
                
            processed_image = cv2.resize(img_array, dim, interpolation=interpolation)
            return Image.fromarray(processed_image)

        elif operation == 'shear':
            shear_factor = args[0]
            M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
            processed_image = cv2.warpAffine(img_array, M, (img_array.shape[1], img_array.shape[0]))

        elif operation == 'laplacian':
            ksize = args[0]
            gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            processed_image = cv2.Laplacian(gray_image, cv2.CV_64F, ksize=ksize)
            processed_image = np.uint8(np.clip(processed_image, 0, 255))
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)

        elif operation == 'gaussian':
            ksize = args[0]
            processed_image = cv2.GaussianBlur(img_array, (ksize, ksize), 0)

        elif operation == 'median':
            ksize = args[0]
            processed_image = cv2.medianBlur(img_array, ksize)

        elif operation == 'canny':
            thresh1, thresh2 = args[0], args[1]
            gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            processed_image = cv2.Canny(gray_image, thresh1, thresh2)
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)

        elif operation == 'convert_color_space':
            color_space = args[0]
            if color_space == 'HSV':
                processed_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            elif color_space == 'LAB':
                processed_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)

        elif operation == 'brightness':
            factor = args[0]
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(factor)

        elif operation == 'contrast':
            factor = args[0]
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(factor)

        # Ensure we have a processed image before conversion
        if processed_image is not None:
            return Image.fromarray(processed_image)
        else:
            return image

    except Exception as e:
        st.error(f"Error processing image: {e}")
        return image

# Streamlit UI
st.title("Image Processing App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    if min(image.size) < 50:
        st.error("Image is too small for processing.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
            plot_histogram(image)

        with st.sidebar:
            st.header("Transformations")
            scale_factor = st.slider("Scale Factor", 0.1, 3.0, 1.0, 0.1)
            # Other transformation parameters...

        # Process and display scaled image
        scaled_img = process_image(image, 'scale', scale_factor)
        
        with col2:
            if scaled_img.size != image.size:  # Only show if actually scaled
                st.image(scaled_img, caption=f"Scaled (Factor: {scale_factor})", use_column_width=True)
                plot_histogram(scaled_img)
            else:
                st.warning("Scale factor is 1.0 - no scaling applied")
