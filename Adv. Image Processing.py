import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Function to plot histogram
def plot_histogram(image):
    img_array = np.array(image.convert('L'))  # Convert to grayscale
    hist_values, _ = np.histogram(img_array.flatten(), bins=256, range=[0, 256])

    fig, ax = plt.subplots()
    ax.plot(hist_values, color='black')
    ax.set_title("Histogram")
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# Image processing function
def process_image(image, operation, *args):
    try:
        img_array = np.array(image)

        if operation == 'scale':
            scale_factor = args[0]
            # Calculate new dimensions while maintaining aspect ratio
            width = int(img_array.shape[1] * scale_factor)
            height = int(img_array.shape[0] * scale_factor)
            dim = (width, height)
            
            # Use INTER_AREA for shrinking and INTER_CUBIC for enlarging
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

        return Image.fromarray(processed_image)

    except Exception as e:
        st.error(f"Error processing image: {e}")
        return image

# Streamlit UI
st.title("Advanced Image Processing App")

if 'processed_images' not in st.session_state:
    st.session_state.processed_images = {}

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    if min(image.size) < 50:
        st.error("Image is too small for processing. Please upload a larger image.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
            plot_histogram(image)

        # Sidebar for transformations
        with st.sidebar:
            st.header("Image Transformations")
            scale_factor = st.slider("Scale Factor", 0.1, 3.0, 1.0, 0.1)
            shear_factor = st.slider("Shear Factor", 0.0, 1.0, 0.0, 0.1)
            laplacian_ksize = st.slider("Laplacian Kernel Size", 1, 7, 1, step=2)
            gaussian_ksize = st.slider("Gaussian Filter Size", 3, 15, 5, step=2)
            median_ksize = st.slider("Median Filter Size", 3, 15, 5, step=2)
            thresh1 = st.slider("Canny Threshold 1", 0, 255, 100)
            thresh2 = st.slider("Canny Threshold 2", 0, 255, 200)
            color_space = st.selectbox("Select Color Space", ["RGB", "HSV", "LAB"])
            brightness = st.slider("Brightness Factor", 0.5, 2.0, 1.0, 0.1)
            contrast = st.slider("Contrast Factor", 0.5, 2.0, 1.0, 0.1)

        # Process images
        scaled_img = process_image(image, 'scale', scale_factor)
        sheared_img = process_image(image, 'shear', shear_factor)
        laplacian_img = process_image(image, 'laplacian', laplacian_ksize)
        gaussian_img = process_image(image, 'gaussian', gaussian_ksize)
        median_img = process_image(image, 'median', median_ksize)
        canny_img = process_image(image, 'canny', thresh1, thresh2)
        color_img = process_image(image, 'convert_color_space', color_space)
        bright_img = process_image(image, 'brightness', brightness)
        contrast_img = process_image(image, 'contrast', contrast)

        # Display scaled image separately for better visibility
        with col2:
            st.image(scaled_img, caption=f"Scaled Image (Factor: {scale_factor})", use_column_width=True)
            plot_histogram(scaled_img)

        # Display other transformations in expanders
        with st.expander("Other Transformations"):
            cols = st.columns(2)
            transformations = [
                ("Shear", sheared_img),
                ("Laplacian", laplacian_img),
                ("Gaussian Blur", gaussian_img),
                ("Median Blur", median_img),
                ("Canny Edge", canny_img),
                (f"{color_space} Color Space", color_img),
                ("Brightness Adjusted", bright_img),
                ("Contrast Adjusted", contrast_img)
            ]
            
            for i, (name, img) in enumerate(transformations):
                with cols[i % 2]:
                    st.image(img, caption=name, use_column_width=True)
                    if name not in ["Shear", "Brightness Adjusted", "Contrast Adjusted"]:
                        plot_histogram(img)

with st.expander("ℹ️ Transformations Explained"):
    st.markdown("""
    - **Scale:** Resizes the image (INTER_AREA for shrinking, INTER_CUBIC for enlarging)
    - **Shear:** Applies a transformation that shifts pixels
    - **Laplacian:** Highlights edges using second-order derivatives
    - **Gaussian:** Blurs the image to reduce noise
    - **Median:** Uses median filtering for noise reduction
    - **Canny Edge Detection:** Detects edges with adjustable thresholds
    - **Color Space Conversion:** Converts between RGB, HSV, and LAB
    - **Brightness & Contrast:** Adjusts intensity levels
    """)
