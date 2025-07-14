import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Load trained model
MODEL_PATH = "aihistograph_model.h5"
CLASS_NAMES = sorted(os.listdir("AiHistrograph_dataset/train"))  # auto-detect class labels

@st.cache_resource
def load_trained_model():
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        return model
    else:
        st.error("Model file not found! Please train and save the model as 'aihistograph_model.h5'.")
        return None

def predict_image(model, img):
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence

# Streamlit app
def main():
    st.title("🖼️ AiHistrography - Image Classification ")
    st.write("Upload an image to classify.")

    model = load_trained_model()
    if model is None:
        return

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        if st.button("Predict"):
            predicted_class, confidence = predict_image(model, img)
            st.success(f"✅ Predicted Class: **{predicted_class}** (Confidence: {confidence:.2%})")

if __name__ == "__main__":
    main()
