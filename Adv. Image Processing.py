import streamlit as st

# Help Section
with st.expander("ℹ️ Visual Guides/Help Section"):
    st.markdown("""
    **Transformations Explained:**
    - **Rotate:** Rotates the image by a specified angle.
    - **Scale:** Resizes the image by a scale factor.
    - **Shear:** Applies a shear transformation to shift image content.
    - **Laplacian:** Enhances edges by applying the Laplacian filter.
    - **Gaussian:** Smooths the image by applying a Gaussian blur.
    - **Median:** Reduces noise by applying a median filter.
    - **Canny Edge Detection:** Detects edges by using the Canny algorithm.
    - **Color Space Conversion:** Converts the image between RGB, HSV, and LAB color spaces.
    - **Brightness:** Adjusts image brightness.
    - **Contrast:** Modifies the contrast to enhance or reduce image details.
    - **Sharpen:** Enhances edges and details by applying a sharpening kernel.
    - **Emboss:** Applies an emboss effect, creating a raised texture.
    """)

st.write("Explore various transformations and visualize their impact on the image!")
