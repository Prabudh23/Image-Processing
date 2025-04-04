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
        processed_image = img_array  # Default to the original image

        if operation == 'scale':
            scale_factor = args[0]
            new_size = (int(img_array.shape[1] * scale_factor), int(img_array.shape[0] * scale_factor))
            processed_image = cv2.resize(img_array, new_size, interpolation=cv2.INTER_LINEAR)

        elif operation == 'shear':
            shear_factor = args[0]
            M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
            processed_image = cv2.warpAffine(img_array, M, (img_array.shape[1], img_array.shape[0]))

        elif operation == 'laplacian':
            ksize = args[0]
            gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            processed_image = cv2.Laplacian(gray_image, cv2.CV_64F, ksize=ksize)
            processed_image = np.uint8(np.clip(processed_image, 0, 255))
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)  # Convert back to RGB

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
        st.image(image, caption="Original Image", use_container_width=True)

        # Sidebar for transformations
        with st.sidebar:
            st.header("Image Transformations")
            actions = {
                "Scale": ("scale", st.slider("Scale Factor", 0.1, 3.0, 1.0)),
                "Shear": ("shear", st.slider("Shear Factor", 0.0, 1.0, 0.0)),
                "Laplacian": ("laplacian", st.slider("Laplacian Kernel Size", 1, 7, 1, step=2)),
                "Gaussian": ("gaussian", st.slider("Gaussian Filter Size", 3, 15, 5, step=2)),
                "Median": ("median", st.slider("Median Filter Size", 3, 15, 5, step=2)),
                "Canny Edge Detection": ("canny", st.slider("Threshold 1", 0, 255, 100), st.slider("Threshold 2", 0, 255, 200)),
                "Color Space Conversion": ("convert_color_space", st.selectbox("Select Color Space", ["RGB", "HSV", "LAB"])),
                "Brightness": ("brightness", st.slider("Brightness Factor", 0.5, 2.0, 1.0)),
                "Contrast": ("contrast", st.slider("Contrast Factor", 0.5, 2.0, 1.0))
            }

            for action, params in actions.items():
                st.session_state.processed_images[action] = process_image(image, *params)

        # Display processed images and histograms for relevant transformations
        st.subheader("Processed Images")
        for name, img in st.session_state.processed_images.items():
            st.image(img, caption=f"{name} Image", use_container_width=True)

            # Only show histogram for transformations that modify intensity values
            if name in ["Laplacian", "Gaussian", "Median", "Canny Edge Detection", "Color Space Conversion"]:
                st.subheader(f"{name} Image Histogram")
                plot_histogram(img)

            # Download button
            img_byte_arr = np.array(img.convert('RGB'))
            is_success, buffer = cv2.imencode(".png", cv2.cvtColor(img_byte_arr, cv2.COLOR_RGB2BGR))
            if is_success:
                if st.download_button(f"Download {name} Image", buffer.tobytes(), file_name=f"{name.lower()}_image.png", mime="image/png"):
                    st.balloons()

with st.expander("ℹ️ Transformations Explained"):
    st.markdown("""
    - **Scale:** Resizes the image.
    - **Shear:** Applies a transformation that shifts pixels.
    - **Laplacian:** Highlights edges using second-order derivatives.
    - **Gaussian:** Blurs the image to reduce noise.
    - **Median:** Uses median filtering for noise reduction.
    - **Canny Edge Detection:** Detects edges with adjustable thresholds.
    - **Color Space Conversion:** Converts between RGB, HSV, and LAB.
    - **Brightness & Contrast:** Adjusts intensity levels.
    """)

st.write("Experiment with different filters and transformations!")
