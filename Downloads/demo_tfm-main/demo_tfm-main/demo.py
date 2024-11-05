import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

st.title('Clasificación de Residuos')

# Cargar el modelo preentrenado
model_path = './best_model.h5'  # Cambia esta ruta al modelo entrenado
model = load_model(model_path)

# Función para cargar y preprocesar la imagen
def load_and_preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalizar la imagen
    return img_array

# Subir una imagen
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar la imagen subida
    img = Image.open(uploaded_file)
    st.image(img, caption='Imagen subida.', use_column_width=True)
    
    # Preprocesar la imagen
    img_array = load_and_preprocess_image(img)
    
    # Hacer predicciones
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    
    # Definir las clases
    class_names = ['battery', 'glass', 'metal', 'organic', 'paper', 'plastic']
    
    # Mostrar la predicción
    st.write(f'Predicción: {class_names[predicted_class[0]]}')