import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance
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
        if operation == 'rotate':
            center = (img_array.shape[1] // 2, img_array.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, args[0], 1.0)
            processed_image = cv2.warpAffine(img_array, M, (img_array.shape[1], img_array.shape[0]))
        elif operation == 'scale':
            new_size = (int(image.width * args[0]), int(image.height * args[0]))
            processed_image = image.resize(new_size, Image.LANCZOS)
            return processed_image
        elif operation == 'shear':
            M = np.float32([[1, args[0], 0], [0, 1, 0]])
            processed_image = cv2.warpAffine(img_array, M, (img_array.shape[1], img_array.shape[0]))
        elif operation == 'laplacian':
            processed_image = cv2.Laplacian(img_array, cv2.CV_64F, ksize=args[0])
            processed_image = np.uint8(np.clip(processed_image, 0, 255))
        elif operation == 'gaussian':
            processed_image = cv2.GaussianBlur(img_array, (args[0], args[0]), 0)
        elif operation == 'median':
            processed_image = cv2.medianBlur(img_array, args[0])
        elif operation == 'bilateral':
            processed_image = cv2.bilateralFilter(img_array, args[0], args[1], args[2])
        elif operation == 'convert_color_space':
            if args[0] == 'HSV':
                processed_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            elif args[0] == 'LAB':
                processed_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        elif operation == 'canny':
            gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            processed_image = cv2.Canny(gray_image, args[0], args[1])
        elif operation == 'brightness':
            enhancer = ImageEnhance.Brightness(image)
            processed_image = enhancer.enhance(args[0])
            return processed_image
        elif operation == 'contrast':
            enhancer = ImageEnhance.Contrast(image)
            processed_image = enhancer.enhance(args[0])
            return processed_image
        elif operation == 'sharpen':
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            processed_image = cv2.filter2D(img_array, -1, kernel)
        elif operation == 'emboss':
            kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            processed_image = cv2.filter2D(img_array, -1, kernel)
        return Image.fromarray(processed_image)
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return image

def annotate_image(image, annotation_type, *args):
    try:
        draw = ImageDraw.Draw(image)
        if annotation_type == 'Rectangle':
            draw.rectangle([args[0], args[1], args[2], args[3]], outline='red', width=3)
        elif annotation_type == 'Circle':
            draw.ellipse([args[0], args[1], args[2], args[3]], outline='blue', width=3)
        elif annotation_type == 'Text':
            draw.text(args[0], args[1], fill='green')
        return image
    except Exception as e:
        st.error(f"Error annotating image: {e}")
        return image

# Streamlit App
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
        st.subheader("Original Image Histogram")
        plot_histogram(image)

        # Sidebar for transformations
        with st.sidebar:
            st.header("Image Transformations")
            actions = {
                "Rotate": ("rotate", st.slider("Rotation Angle", 0, 360, 0)),
                "Scale": ("scale", st.slider("Scale Factor", 0.1, 3.0, 1.0)),
                "Shear": ("shear", st.slider("Shear Factor", 0.0, 1.0, 0.0)),
                "Laplacian": ("laplacian", st.slider("Laplacian Kernel Size", 1, 7, 1, step=2)),
                "Gaussian": ("gaussian", st.slider("Gaussian Filter Size", 3, 15, 5, step=2)),
                "Median": ("median", st.slider("Median Filter Size", 3, 15, 5, step=2)),
                "Canny Edge Detection": ("canny", st.slider("Threshold 1", 0, 255, 100), st.slider("Threshold 2", 0, 255, 200)),
                "Color Space Conversion": ("convert_color_space", st.selectbox("Select Color Space", ["RGB", "HSV", "LAB"])),
                "Brightness": ("brightness", st.slider("Brightness Factor", 0.5, 2.0, 1.0)),
                "Contrast": ("contrast", st.slider("Contrast Factor", 0.5, 2.0, 1.0)),
                "Sharpen": ("sharpen",),
                "Emboss": ("emboss",)
            }

            for action, params in actions.items():
                st.session_state.processed_images[action] = process_image(image, *params)

        # Display processed images
        st.subheader("Processed Images")
        for name, img in st.session_state.processed_images.items():
            st.image(img, caption=f"{name} Image", use_container_width=True)
            st.subheader(f"{name} Image Histogram")
            plot_histogram(img)

            # Download button
            img_byte_arr = np.array(img.convert('RGB'))
            is_success, buffer = cv2.imencode(".png", cv2.cvtColor(img_byte_arr, cv2.COLOR_RGB2BGR))
            if is_success:
                st.download_button(f"Download {name} Image", buffer.tobytes(), file_name=f"{name.lower()}_image.png", mime="image/png")
