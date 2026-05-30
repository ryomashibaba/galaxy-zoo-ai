import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = 128

model = tf.keras.models.load_model("galaxy_model.keras")

def predict_galaxy(image):
    image = image.convert("RGB")
    resized = image.resize((IMG_SIZE, IMG_SIZE))

    x = np.array(resized) / 255.0
    x = x[None, ...]

    pred = model.predict(x, verbose=0)[0][0]

    if pred > 0.5:
        label = "Spiral / Disk Galaxy"
        confidence = pred
    else:
        label = "Smooth / Elliptical Galaxy"
        confidence = 1 - pred

    return label, f"{confidence * 100:.2f}%"

demo = gr.Interface(
    fn=predict_galaxy,
    inputs=gr.Image(type="pil", label="Upload a galaxy image"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Confidence")
    ],
    title="Galaxy Zoo AI",
    description="Upload a galaxy image. The AI predicts whether it is Spiral/Disk or Smooth/Elliptical."
)

demo.launch()
