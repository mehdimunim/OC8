import streamlit as st
import requests
import base64
from PIL import Image
import io
import numpy as np
import cv2

# URL de votre API Flask
API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Segmentation d'Image", layout="wide")

st.title("Application de Segmentation d'Image")

uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image originale", use_column_width=True)

    if st.button("Segmenter l'image"):
        # Préparer l'image pour l'envoi
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Faire la requête à l'API
        with st.spinner("Segmentation en cours..."):
            try:
                response = requests.post(API_URL, files={"image": uploaded_file})
                response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP
            except requests.exceptions.RequestException as e:
                st.error(f"Erreur lors de la communication avec l'API: {e}")
            else:
                if response.status_code == 200:
                    # Traiter la réponse
                    mask_base64 = response.json()['mask']
                    mask_image = Image.open(io.BytesIO(base64.b64decode(mask_base64)))

                    # Afficher l'image originale et le masque côte à côte
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Image originale", use_column_width=True)
                    with col2:
                        st.image(mask_image, caption="Masque segmenté", use_column_width=True)

                    # Superposer le masque sur l'image originale
                    mask_array = np.array(mask_image)
                    colored_mask = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
                    colored_mask[mask_array > 0] = [255, 0, 0]  # Rouge pour les zones segmentées

                    original_array = np.array(image.resize(mask_image.size))
                    overlay = Image.fromarray(cv2.addWeighted(original_array, 0.7, colored_mask, 0.3, 0))

                    st.image(overlay, caption="Superposition", use_column_width=True)
                else:
                    st.error("Erreur lors de la segmentation de l'image")

st.sidebar.title("À propos")
st.sidebar.info("Cette application utilise un modèle de deep learning pour segmenter les images.")

