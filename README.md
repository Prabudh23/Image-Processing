# 🖼️ Image Transformation Tool

This is a user-friendly **Streamlit web app** that allows users to apply various image transformation techniques on uploaded images. Users can rotate, shear, scale, convert to grayscale, apply Laplacian filters, add different types of noise, invert colors, and adjust brightness — all from an interactive sidebar.

## 🚀 Features

- ✅ Image Upload (JPEG, JPG, PNG)
- 🔄 Rotate Image (−180° to 180°)
- ↘️ Shear Image (X and Y axes)
- 🔍 Scale Image (Zoom in/out)
- 🎨 Convert to Grayscale
- 🧠 Apply Laplacian Edge Detection
- 🌫️ Add Noise (Gaussian or Rayleigh)
- 🔁 Invert Colors
- 🔆 Brightness Control (−100 to +100)
- 💾 Download Transformed Images
- 🎉 Visual Feedback on Successful Downloads

## 📦 Requirements

- Python 3.7+
- Streamlit
- OpenCV (`cv2`)
- NumPy
- PIL (Pillow)

Install the dependencies using pip:

```bash
pip install streamlit opencv-python-headless numpy pillow
