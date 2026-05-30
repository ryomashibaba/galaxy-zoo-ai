import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = 128

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("galaxy_model.keras")

model = load_model()

st.title("Galaxy Zoo AI")

uploaded = st.file_uploader(
    "Upload Galaxy Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:

    img = Image.open(uploaded).convert("RGB")

    st.image(img)

    x = img.resize((IMG_SIZE, IMG_SIZE))
    x = np.array(x) / 255.0
    x = x[None, ...]

    pred = model.predict(x, verbose=0)[0][0]

    if pred > 0.5:
        st.success("Spiral / Disk Galaxy")
    else:
        st.info("Smooth / Elliptical Galaxy")

    st.write("Probability:", float(pred))