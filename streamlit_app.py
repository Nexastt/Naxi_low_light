import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras
import io

# ----------------------------
# Load the model
# ----------------------------
@st.cache_resource
def load_model():
    model = keras.models.load_model(
        "naxi_lowlight.keras",
        custom_objects={
            "charbonnier_loss": lambda y_true, y_pred: tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + 1e-6)),
            "psnr_metric": lambda y_true, y_pred: tf.image.psnr(y_pred, y_true, max_val=1.0)
        }
    )
    return model

model = load_model()

# ----------------------------
# Inference function
# ----------------------------
def enhance_image_pil(pil_img):
    image = keras.utils.img_to_array(pil_img).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    output = model.predict(image)[0]
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(output)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="NaxiLowLight Enhancement", layout="centered")
st.title("🌙 NaxiLowLight: Low-Light Image Enhancer")

st.write("Upload a low-light image to enhance it using a deep learning model.")

uploaded_file = st.file_uploader("Choose a low-light image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")

    # Enhance
    st.write("✨ Enhancing image...")
    enhanced_image = enhance_image_pil(image)
    autocontrast_image = ImageOps.autocontrast(image)

    # Show results
    st.write("📷 **Comparison:**")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image, caption="Original", use_container_width=True)

    with col2:
        st.image(autocontrast_image, caption="Autocontrast", use_container_width=True)

    with col3:
        st.image(enhanced_image, caption="NaxiLowLight Enhanced", use_container_width=True)

    # Download
    buf = io.BytesIO()
    enhanced_image.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="⬇️ Download Enhanced Image",
        data=byte_im,
        file_name="enhanced_image.png",
        mime="image/png"
    )
else:
    st.info("Upload a PNG or JPG image to get started.")
