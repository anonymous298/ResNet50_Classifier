import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

file = st.file_uploader('Choose an Image File', type=['jpg', 'png'])

if file:

    img = load_img(file, target_size=(224, 224, 3))
    st.image(img)

    X = img_to_array(img)
    X = np.expand_dims(X, axis=0)
    X = preprocess_input(X)

    prediction = model.predict(X)
    classes = decode_predictions(prediction, top=5)[0]
    final_predictions = [i[1] for i in classes]
    st.write(final_predictions)