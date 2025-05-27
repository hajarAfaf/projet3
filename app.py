
%%writefile app.py

import streamlit as st
import tempfile
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import kagglehub


IMG_SIZE = (64, 64)
st.set_page_config(
    page_title="Classification Astronomique",
    page_icon="🔭",
    layout="wide"
)

# Fonction pour charger le modèle (avec cache pour ne pas recharger à chaque interaction)
@st.cache_resource
def load_my_model():
    if os.path.exists("mon_modele.keras"):
        return tf.keras.models.load_model("mon_modele.keras")
    return None
model = load_my_model()

st.title("🌌 Astronomy Object Classifier")
st.write("Upload an image of a star or galaxy to classify it.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Charger l'image
       img = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=IMG_SIZE)
       img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
       img_expanded = np.expand_dims(img_array, axis=0)
       col1, col2 = st.columns(2)
       with col1:
            st.image(img, caption="Uploaded Image", use_container_width=True)
       with col2:
        if model:
            prediction = model.predict(img_expanded)[0]
            class_names = ["Star", "Galaxy"]
            emojis = ["⭐", "🌌"]
            predicted_index = np.argmax(prediction)
            predicted_class = class_names[predicted_index]
            emoji = emojis[predicted_index]
            confidence = np.max(prediction) * 100

            st.success("**Prediction**")
            st.markdown(f"**Class:** {predicted_class} {emoji}")
            st.markdown(f"**Confidence:** {confidence:.2f}%")
        else:
            st.error(" ⚠️ Model not found. Please make sure 'mon_model.keras' exists.")


