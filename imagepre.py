import streamlit as st  
import cv2  
import numpy as np  
from PIL import Image  
import io  

st.set_page_config(page_title="Image Transformation Tool", layout="centered")  

st.title("Image Transformation Filters")  
st.markdown("Apply **rotation**, **shear**, **scaling**, and **Laplacian filtering** on your uploaded image.")  

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])  

if uploaded_file:  
    image = Image.open(uploaded_file).convert("RGB")  
    img_np = np.array(image)  
    h, w = img_np.shape[:2]  
    center = (w // 2, h // 2)  

    # Display original image  
    st.image(image, caption="Original Image", use_column_width=True)  

    # Sidebar controls  
    st.sidebar.header("🔧 Transformation Settings")  
    rotation_angle = st.sidebar.slider("Rotate (degrees)", -180, 180, 0)  
    shear_x = st.sidebar.slider("Shear X", -1.0, 1.0, 0.0, step=0.01)  
    shear_y = st.sidebar.slider("Shear Y", -1.0, 1.0, 0.0, step=0.01)  
    scale_factor = st.sidebar.slider("Scale Factor", 0.1, 3.0, 1.0, step=0.1)  
    apply_laplacian = st.sidebar.checkbox("Apply Laplacian Filter", value=False)  
    laplacian_size = st.sidebar.selectbox("Laplacian Kernel Size", [3, 5]) if apply_laplacian else None  

    # --- Rotation  
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)  
    rotated_img = cv2.warpAffine(img_np, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)  

    # --- Shearing  
    shear_matrix = np.array([  
        [1, shear_x, 0],  
        [shear_y, 1, 0]  
    ], dtype=np.float32)  
    sheared_img = cv2.warpAffine(img_np, shear_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)  

    # --- Scaling (centered with padding to original size)  
    scaled_raw = cv2.resize(img_np, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)  
    sh, sw = scaled_raw.shape[:2]  
    scaled_img = np.zeros_like(img_np)  

    y_offset = max((h - sh) // 2, 0)  
    x_offset = max((w - sw) // 2, 0)  

    y1 = y_offset  
    y2 = y_offset + min(sh, h)  
    x1 = x_offset  
    x2 = x_offset + min(sw, w)  

    scaled_img[y1:y2, x1:x2] = scaled_raw[0:(y2 - y1), 0:(x2 - x1)]  

    # --- Laplacian  
    if apply_laplacian:  
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)  
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=laplacian_size)  
        lap = cv2.convertScaleAbs(lap)  
        laplacian_img = cv2.cvtColor(lap, cv2.COLOR_GRAY2RGB)  
    
    # Function to convert images to bytes for download  
    def convert_to_bytes(image: np.ndarray, fmt: str = "PNG") -> bytes:  
        _, buf = cv2.imencode(f".{fmt.lower()}", image)  
        return buf.tobytes()  

    # Show transformed images  
    st.subheader("Rotated Image")  
    st.image(rotated_img, use_column_width=True)  
    st.download_button("Download Rotated Image", convert_to_bytes(rotated_img), "rotated_image.png", "image/png")  

    st.subheader("Sheared Image")  
    st.image(sheared_img, use_column_width=True)  
    st.download_button("Download Sheared Image", convert_to_bytes(sheared_img), "sheared_image.png", "image/png")  

    st.subheader("Scaled Image")  
    st.image(scaled_img, use_column_width=True)  
    st.download_button("Download Scaled Image", convert_to_bytes(scaled_img), "scaled_image.png", "image/png")  

    if apply_laplacian:  
        st.subheader(f"Laplacian Filtered Image (Kernel: {laplacian_size}x{laplacian_size})")  
        st.image(laplacian_img, use_column_width=True)  
        st.download_button("Download Laplacian Filtered Image", convert_to_bytes(laplacian_img), "laplacian_image.png", "image/png")  

    st.success("All transformations applied successfully!")  

else:  
    st.info("Please upload an image to begin.")  import streamlit as st  
import cv2  
import numpy as np  
from PIL import Image  
import io  

st.set_page_config(page_title="Image Transformation Tool", layout="centered")  

st.title("Image Transformation Filters")  
st.markdown("Apply **rotation**, **shear**, **scaling**, and **Laplacian filtering** on your uploaded image.")  

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])  

if uploaded_file:  
    image = Image.open(uploaded_file).convert("RGB")  
    img_np = np.array(image)  
    h, w = img_np.shape[:2]  
    center = (w // 2, h // 2)  

    # Display original image  
    st.image(image, caption="Original Image", use_column_width=True)  

    # Sidebar controls  
    st.sidebar.header("🔧 Transformation Settings")  
    rotation_angle = st.sidebar.slider("Rotate (degrees)", -180, 180, 0)  
    shear_x = st.sidebar.slider("Shear X", -1.0, 1.0, 0.0, step=0.01)  
    shear_y = st.sidebar.slider("Shear Y", -1.0, 1.0, 0.0, step=0.01)  
    scale_factor = st.sidebar.slider("Scale Factor", 0.1, 3.0, 1.0, step=0.1)  
    apply_laplacian = st.sidebar.checkbox("Apply Laplacian Filter", value=False)  
    laplacian_size = st.sidebar.selectbox("Laplacian Kernel Size", [3, 5]) if apply_laplacian else None  

    # --- Rotation  
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)  
    rotated_img = cv2.warpAffine(img_np, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)  

    # --- Shearing  
    shear_matrix = np.array([  
        [1, shear_x, 0],  
        [shear_y, 1, 0]  
    ], dtype=np.float32)  
    sheared_img = cv2.warpAffine(img_np, shear_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)  

    # --- Scaling (centered with padding to original size)  
    scaled_raw = cv2.resize(img_np, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)  
    sh, sw = scaled_raw.shape[:2]  
    scaled_img = np.zeros_like(img_np)  

    y_offset = max((h - sh) // 2, 0)  
    x_offset = max((w - sw) // 2, 0)  

    y1 = y_offset  
    y2 = y_offset + min(sh, h)  
    x1 = x_offset  
    x2 = x_offset + min(sw, w)  

    scaled_img[y1:y2, x1:x2] = scaled_raw[0:(y2 - y1), 0:(x2 - x1)]  

    # --- Laplacian  
    if apply_laplacian:  
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)  
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=laplacian_size)  
        lap = cv2.convertScaleAbs(lap)  
        laplacian_img = cv2.cvtColor(lap, cv2.COLOR_GRAY2RGB)  
    
    # Function to convert images to bytes for download  
    def convert_to_bytes(image: np.ndarray, fmt: str = "PNG") -> bytes:  
        _, buf = cv2.imencode(f".{fmt.lower()}", image)  
        return buf.tobytes()  

    # Show transformed images  
    st.subheader("Rotated Image")  
    st.image(rotated_img, use_column_width=True)  
    st.download_button("Download Rotated Image", convert_to_bytes(rotated_img), "rotated_image.png", "image/png")  

    st.subheader("Sheared Image")  
    st.image(sheared_img, use_column_width=True)  
    st.download_button("Download Sheared Image", convert_to_bytes(sheared_img), "sheared_image.png", "image/png")  

    st.subheader("Scaled Image")  
    st.image(scaled_img, use_column_width=True)  
    st.download_button("Download Scaled Image", convert_to_bytes(scaled_img), "scaled_image.png", "image/png")  

    if apply_laplacian:  
        st.subheader(f"Laplacian Filtered Image (Kernel: {laplacian_size}x{laplacian_size})")  
        st.image(laplacian_img, use_column_width=True)  
        st.download_button("Download Laplacian Filtered Image", convert_to_bytes(laplacian_img), "laplacian_image.png", "image/png")  

    st.success("All transformations applied successfully!")  

else:  
    st.info("Please upload an image to begin.")  
