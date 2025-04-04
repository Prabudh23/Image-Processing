import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Image Processor", page_icon="🖼️", layout="wide")

# Custom CSS for better appearance
st.markdown("""
    <style>
    .main {background-color: #f9f9f9;}
    .stButton>button {border-radius: 5px; padding: 8px 20px;}
    .stDownloadButton>button {background-color: #4CAF50; color: white;}
    .stFileUploader>div>div>div>div {border: 2px dashed #4CAF50;}
    .sidebar .sidebar-content {background-color: #e8f5e9;}
    .image-container {margin-bottom: 20px;}
    .image-title {font-weight: bold; text-align: center;}
    .param-slider {margin-bottom: 15px;}
    </style>
    """, unsafe_allow_html=True)

def plot_histogram(img):
    if len(img.shape) == 2:  # Grayscale
        plt.figure(figsize=(8, 4))
        plt.hist(img.ravel(), 256, [0, 256], color='gray')
        plt.title('Grayscale Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
    else:  # Color
        plt.figure(figsize=(8, 4))
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)
        plt.title('Color Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
    return plt

def shear_image(img, shear_factor, axis):
    if axis == 'x':
        shear_matrix = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    else:
        shear_matrix = np.float32([[1, 0, 0], [shear_factor, 1, 0]])
    
    rows, cols = img.shape[:2]
    sheared_img = cv2.warpAffine(img, shear_matrix, (int(cols + abs(shear_factor)*rows), rows))
    return sheared_img

def apply_operation(img, operation, params):
    try:
        if operation == "Original":
            return img
        elif operation == "Grayscale":
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif operation == "Blur":
            ksize = params['blur_kernel']
            return cv2.GaussianBlur(img, (ksize, ksize), 0)
        elif operation == "Edge Detection":
            threshold1 = params['edge_threshold1']
            threshold2 = params['edge_threshold2']
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.Canny(gray, threshold1, threshold2)
        elif operation == "Threshold":
            thresh_value = params['threshold_value']
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
            return thresh
        elif operation == "Contrast Stretch":
            clip_limit = params['clip_limit']
            grid_size = params['grid_size']
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        elif operation == "Negative":
            return 255 - img
        elif operation == "Sepia":
            intensity = params['sepia_intensity']
            kernel = np.array([[0.272*intensity, 0.534*intensity, 0.131*intensity],
                              [0.349*intensity, 0.686*intensity, 0.168*intensity],
                              [0.393*intensity, 0.769*intensity, 0.189*intensity]])
            return cv2.filter2D(img, -1, kernel)
        elif operation == "Sketch":
            kernel_size = params['sketch_kernel']
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            inv_gray = 255 - gray
            blurred = cv2.GaussianBlur(inv_gray, (kernel_size, kernel_size), 0)
            inv_blurred = 255 - blurred
            return cv2.divide(gray, inv_blurred, scale=256.0)
        elif operation == "Emboss":
            intensity = params['emboss_intensity']
            kernel = np.array([[0, -1*intensity, -1*intensity],
                              [1*intensity, 0, -1*intensity],
                              [1*intensity, 1*intensity, 0]])
            return cv2.filter2D(img, -1, kernel)
        return img
    except Exception as e:
        st.error(f"Error applying {operation}: {str(e)}")
        return img

def display_image_with_title(img, title):
    col1, col2, col3 = st.columns([1,6,1])
    with col2:
        st.markdown(f'<p class="image-title">{title}</p>', unsafe_allow_html=True)
        st.image(img, use_column_width=True)

def get_operation_params(operation):
    params = {}
    if operation == "Blur":
        params['blur_kernel'] = st.slider("Blur Kernel Size", 1, 31, 15, 2, key="blur_kernel")
    elif operation == "Edge Detection":
        params['edge_threshold1'] = st.slider("Threshold 1", 1, 255, 100, key="edge_threshold1")
        params['edge_threshold2'] = st.slider("Threshold 2", 1, 255, 200, key="edge_threshold2")
    elif operation == "Threshold":
        params['threshold_value'] = st.slider("Threshold Value", 0, 255, 127, key="threshold_value")
    elif operation == "Contrast Stretch":
        params['clip_limit'] = st.slider("Clip Limit", 1.0, 10.0, 3.0, 0.1, key="clip_limit")
        params['grid_size'] = st.slider("Grid Size", 2, 16, 8, 2, key="grid_size")
    elif operation == "Sepia":
        params['sepia_intensity'] = st.slider("Sepia Intensity", 0.1, 2.0, 1.0, 0.1, key="sepia_intensity")
    elif operation == "Sketch":
        params['sketch_kernel'] = st.slider("Sketch Kernel Size", 1, 31, 21, 2, key="sketch_kernel")
    elif operation == "Emboss":
        params['emboss_intensity'] = st.slider("Emboss Intensity", 0.1, 2.0, 1.0, 0.1, key="emboss_intensity")
    elif operation == "Shear":
        params['shear_axis'] = st.radio("Shear Axis", ['x', 'y'], key="shear_axis")
        params['shear_factor'] = st.slider("Shear Factor", -1.0, 1.0, 0.2, 0.1, key="shear_factor")
    return params

def main():
    st.title("🖼️ Image Processing App")
    st.markdown("Upload an image to see different filter effects")
    
    # Sidebar for upload
    with st.sidebar:
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Main content area
    if uploaded_file is not None:
        # Read and convert image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Display original image and histogram
        display_image_with_title(img, "Original Image")
        
        st.subheader("Histogram")
        fig = plot_histogram(img)
        st.pyplot(fig)
        
        st.divider()
        st.subheader("Filter Effects")
        
        # Define all operations
        operations = [
            "Original",
            "Grayscale",
            "Blur",
            "Edge Detection",
            "Threshold",
            "Contrast Stretch",
            "Negative",
            "Sepia",
            "Sketch",
            "Emboss",
            "Shear"
        ]
        
        # Operation selection and parameters
        selected_operation = st.selectbox("Select Operation:", operations)
        
        st.subheader(f"{selected_operation} Parameters")
        params = get_operation_params(selected_operation)
        
        # Apply operation
        processed_img = apply_operation(img, selected_operation, params)
        
        # Display processed image
        display_image_with_title(processed_img, f"{selected_operation} Effect")
        
        # Download section
        st.divider()
        st.subheader("Download Processed Image")
        
        processed_pil = Image.fromarray(processed_img)
        buf = io.BytesIO()
        processed_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        if st.download_button(
            label="⬇️ Download Processed Image",
            data=byte_im,
            file_name=f"processed_{selected_operation.lower().replace(' ', '_')}.png",
            mime="image/png"
        ):
            st.balloons()
            st.success("Download started!")
    
    else:
        st.info("Please upload an image to get started")

if __name__ == "__main__":
    main()
