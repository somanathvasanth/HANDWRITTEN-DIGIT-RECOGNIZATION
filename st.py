import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from PIL import Image

# Load trained MNIST model
model = load_model('mnist.h5')

st.title("Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (MNIST-style, grayscale)")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert('L')  # convert to grayscale
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess image
    img = np.array(image)
    img = cv2.resize(img, (28, 28))
    img = 255 - img  # invert colors if needed
    img = img / 255.0  # normalize
    img = img.reshape(1, 28, 28, 1)
    
    # Prediction
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)
    
    st.subheader(f'Predicted Digit: {predicted_digit}')
