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

def apply_operation(img, operation):
    if operation == "Original":
        return img
    elif operation == "Grayscale":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif operation == "Blur":
        return cv2.GaussianBlur(img, (15, 15), 0)
    elif operation == "Edge Detection":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, 100, 200)
    elif operation == "Threshold":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return thresh
    elif operation == "Contrast Stretch":
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    elif operation == "Negative":
        return 255 - img
    elif operation == "Sepia":
        kernel = np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])
        return cv2.filter2D(img, -1, kernel)
    elif operation == "Sketch":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv_gray = 255 - gray
        blurred = cv2.GaussianBlur(inv_gray, (21, 21), 0)
        inv_blurred = 255 - blurred
        return cv2.divide(gray, inv_blurred, scale=256.0)
    elif operation == "Emboss":
        kernel = np.array([[0, -1, -1],
                          [1,  0, -1],
                          [1,  1,  0]])
        return cv2.filter2D(img, -1, kernel)
    elif operation == "Oil Painting":
        return cv2.xphoto.oilPainting(img, 7, 1)
    return img

def display_image_with_title(img, title):
    col1, col2, col3 = st.columns([1,6,1])
    with col2:
        st.markdown(f'<p class="image-title">{title}</p>', unsafe_allow_html=True)
        st.image(img, use_column_width=True)

def main():
    st.title("🖼️ Image Processing App")
    st.markdown("Upload an image to see different filter effects")
    
    # Sidebar for upload and operations
    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            st.subheader("Shearing Options")
            shear_axis = st.radio("Shear Axis", ['x', 'y'])
            shear_factor = st.slider("Shear Factor", -1.0, 1.0, 0.2, 0.1)
    
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
            "Oil Painting"
        ]
        
        # Display all filtered images in a grid
        cols = st.columns(3)
        for i, operation in enumerate(operations[1:]):  # Skip "Original"
            processed_img = apply_operation(img, operation)
            processed_img = shear_image(processed_img, shear_factor, shear_axis)
            
            with cols[i % 3]:
                st.markdown(f'<p class="image-title">{operation}</p>', unsafe_allow_html=True)
                st.image(processed_img, use_column_width=True)
        
        # Download section
        st.divider()
        st.subheader("Download Processed Image")
        
        selected_operation = st.selectbox("Choose an operation to download:", operations)
        processed_img = apply_operation(img, selected_operation)
        processed_img = shear_image(processed_img, shear_factor, shear_axis)
        
        display_image_with_title(processed_img, f"Selected for Download: {selected_operation}")
        
        # Download button
        processed_pil = Image.fromarray(processed_img)
        buf = io.BytesIO()
        processed_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        if st.download_button(
            label="⬇️ Download Image",
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
