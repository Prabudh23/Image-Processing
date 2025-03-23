import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
import cv2
import matplotlib.pyplot as plt
from fpdf import FPDF
import io

def plot_histogram(image):
    img_array = np.array(image.convert('L'))
    hist_values, _ = np.histogram(img_array.flatten(), bins=256, range=[0, 256])
    fig, ax = plt.subplots()
    ax.plot(hist_values, color='black')
    ax.set_title("Histogram")
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf.read()

def generate_pdf_report(original_image, processed_images):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Image Processing Report", ln=True, align='C')
    pdf.ln(10)
    
    # Original Image Section
    pdf.cell(0, 10, "Original Image:", ln=True)
    original_path = "original_image.png"
    original_image.save(original_path)
    pdf.image(original_path, x=10, w=100)
    
    # Original Histogram
    pdf.ln(5)
    pdf.cell(0, 10, "Original Histogram:", ln=True)
    original_hist = plot_histogram(original_image)
    original_hist_path = "original_hist.png"
    with open(original_hist_path, "wb") as f:
        f.write(original_hist)
    pdf.image(original_hist_path, x=10, w=100)
    pdf.ln(10)

    # Processed Images Section
    for name, img in processed_images.items():
        pdf.cell(0, 10, f"{name} Image:", ln=True)
        processed_path = f"{name.lower().replace(' ', '_')}_image.png"
        img.save(processed_path)
        pdf.image(processed_path, x=10, w=100)
        
        # Processed Histogram
        pdf.ln(5)
        pdf.cell(0, 10, f"{name} Histogram:", ln=True)
        processed_hist = plot_histogram(img)
        processed_hist_path = f"{name.lower().replace(' ', '_')}_hist.png"
        with open(processed_hist_path, "wb") as f:
            f.write(processed_hist)
        pdf.image(processed_hist_path, x=10, w=100)
        pdf.ln(10)

    # Save PDF
    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output

# Streamlit App
st.title("Advanced Image Processing App with PDF Report")
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
                "Brightness": ("brightness", st.slider("Brightness Factor", 0.5, 2.0, 1.0)),
                "Contrast": ("contrast", st.slider("Contrast Factor", 0.5, 2.0, 1.0)),
                "Sharpen": ("sharpen",),
                "Emboss": ("emboss",)
            }

            for action, params in actions.items():
                st.session_state.processed_images[action] = image  # Placeholder for processed image logic

        st.subheader("Processed Images")
        for name, img in st.session_state.processed_images.items():
            st.image(img, caption=f"{name} Image", use_container_width=True)
            st.subheader(f"{name} Image Histogram")
            plot_histogram(img)

        st.subheader("Generate PDF Report")
        pdf_output = generate_pdf_report(image, st.session_state.processed_images)
        st.download_button("Download PDF Report", pdf_output, file_name="image_processing_report.pdf", mime="application/pdf")
