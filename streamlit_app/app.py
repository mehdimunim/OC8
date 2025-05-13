"""
Application de Segmentation d'Images de Voitures Autonomes

Cette application Streamlit permet de:
1. Télécharger des images de scènes urbaines
2. Visualiser leur segmentation (prédite par une API)
3. Comparer avec un masque réel (optionnel)

L'application s'intègre avec une API Flask qui effectue la segmentation d'image
en utilisant un modèle de deep learning entraîné sur le dataset Cityscapes.
"""

import streamlit as st
import requests
from PIL import Image
import numpy as np
import io
import os
import cv2
import base64

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

# URL de l'API de segmentation
API_URL = "http://localhost:5000/predict"

# Nombre de classes de segmentation
NUM_CLASSES = 8

# Mapping des classes aux noms d'objets et aux couleurs pour la visualisation
CLASS_MAPPING_COLORS = {
    0: ("void", [0, 0, 0]),              # Noir
    1: ("flat", [255, 255, 0]),          # Jaune
    2: ("human", [255, 0, 255]),         # Magenta
    3: ("vehicle", [0, 0, 255]),         # Bleu
    4: ("construction", [0, 255, 0]),    # Vert
    5: ("object", [255, 0, 0]),          # Rouge
    6: ("nature", [0, 255, 255]),        # Cyan
    7: ("sky", [255, 255, 255])          # Blanc
}

# ------------------------------------------------------------------------------
# Fonctions utilitaires
# ------------------------------------------------------------------------------

def create_legend(class_mapping_colors):
    """
    Crée une légende HTML pour visualiser les classes et leurs couleurs associées.
    
    Args:
        class_mapping_colors (dict): Dictionnaire associant indices de classe à (nom, couleur RGB)
    
    Returns:
        tuple: (legend_html, color_map) 
            - legend_html: Code HTML de la légende à afficher
            - color_map: Dictionnaire associant indices de classe à couleurs RGB
    """
    legend_items = []
    color_map = {}
    
    for class_index, (label, color) in class_mapping_colors.items():
        color_hex = '#%02x%02x%02x' % tuple(color)
        legend_items.append(
            f'<span style="background-color:{color_hex}; color:{color_hex}; '
            f'padding: 5px; border-radius: 3px;">&nbsp;&nbsp;&nbsp;</span> {label}'
        )
        color_map[class_index] = color
        
    return "<br>".join(legend_items), color_map


def colorize_mask(mask, color_map):
    """
    Transforme un masque en niveaux de gris (indices de classe) en une image colorée.
    
    Args:
        mask (ndarray): Masque de segmentation (H,W) avec des valeurs d'indices de classe
        color_map (dict): Dictionnaire associant indices de classe à couleurs RGB
    
    Returns:
        PIL.Image: Image colorée où chaque classe a sa couleur spécifique
    """
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    for class_index, color in color_map.items():
        mask_indices = (mask == class_index)
        if np.any(mask_indices):  # Vérifie si la classe existe dans le masque
            colored_mask[mask_indices] = color
            
    return Image.fromarray(colored_mask)


def overlay_mask(image, mask, colors, alpha=0.5):
    """
    Superpose un masque coloré sur une image.
    
    Args:
        image (PIL.Image): Image d'origine
        mask (PIL.Image): Masque coloré
        colors (dict): Dictionnaire de couleurs (non utilisé directement mais conservé pour cohérence d'API)
        alpha (float): Niveau de transparence du masque (0.0 à 1.0)
    
    Returns:
        PIL.Image: Image avec le masque superposé
    """
    img = np.array(image, dtype=np.uint8)
    mask_img = np.array(mask.convert("RGB"), dtype=np.uint8)
    
    # S'assurer que les dimensions correspondent
    if img.shape[:2] != mask_img.shape[:2]:
        mask_img = cv2.resize(mask_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    masked_image = cv2.addWeighted(img, 1 - alpha, mask_img, alpha, 0)
    return Image.fromarray(masked_image)


def predict_mask(image):
    """
    Envoie une image à l'API de segmentation et récupère le masque prédit.
    
    Args:
        image (PIL.Image): Image à segmenter
    
    Returns:
        tuple: (success, result)
            - success (bool): Indique si la prédiction a réussi
            - result: Image du masque prédit si success est True, message d'erreur sinon
    """
    try:
        # Préparer l'image pour l'envoi à l'API
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        
        # Appeler l'API
        response = requests.post(
            API_URL, 
            files={"image": ("image.png", img_bytes, "image/png")}
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Vérifier et décoder la réponse base64
            mask_base64 = data.get('mask')
            if not mask_base64:
                return False, "La réponse de l'API ne contient pas le champ 'mask' attendu."
                
            mask_bytes = base64.b64decode(mask_base64)
            predicted_mask = Image.open(io.BytesIO(mask_bytes))
            return True, predicted_mask
        else:
            return False, f"Erreur de l'API (code {response.status_code}): {response.text}"
    
    except requests.exceptions.RequestException as e:
        return False, f"Erreur lors de la communication avec l'API: {e}"
    except UnicodeEncodeError as e:
        return False, f"Erreur d'encodage: {e}"
    except Exception as e:
        return False, f"Une erreur inattendue s'est produite: {e}"


def process_predicted_mask(predicted_mask_img):
    """
    Traite le masque prédit par l'API pour s'assurer qu'il est au bon format.
    
    Args:
        predicted_mask_img (PIL.Image): Masque prédit par l'API
    
    Returns:
        ndarray: Masque traité comme tableau d'indices de classe
    """
    predicted_mask_np = np.array(predicted_mask_img)
    
    # Si le masque est multi-canal (probabilités), le convertir en indices de classe
    if predicted_mask_np.ndim == 3 and predicted_mask_np.shape[2] > 1:
        predicted_mask_np = np.argmax(predicted_mask_np, axis=-1)
        st.info("Le masque prédit semblait être une probabilité, converti en indices de classe.")
        
    return predicted_mask_np


def compare_masks(real_mask_np, predicted_mask_np):
    """
    Compare un masque prédit avec un masque réel et calcule des métriques.
    
    Args:
        real_mask_np (ndarray): Masque réel
        predicted_mask_np (ndarray): Masque prédit
    
    Returns:
        tuple: (accuracy, resized_real_mask)
            - accuracy (float): Précision pixel par pixel (pourcentage)
            - resized_real_mask (ndarray): Masque réel redimensionné si nécessaire
    """
    # Redimensionner si nécessaire
    if real_mask_np.shape != predicted_mask_np.shape:
        real_mask_np = cv2.resize(
            real_mask_np, 
            (predicted_mask_np.shape[1], predicted_mask_np.shape[0]), 
            interpolation=cv2.INTER_NEAREST
        )
        st.warning("Les dimensions des masques ont été ajustées pour la comparaison.")
    
    # Calcul de la précision pixel par pixel
    accuracy = np.mean(real_mask_np == predicted_mask_np) * 100
    
    return accuracy, real_mask_np


# ------------------------------------------------------------------------------
# Interface utilisateur Streamlit
# ------------------------------------------------------------------------------

def main():
    """
    Fonction principale qui définit l'interface utilisateur de l'application Streamlit.
    """
    # Titre et sous-titre de l'application
    st.title("Application de Segmentation d'Images de Voitures")
    st.subheader("Visualisation de la segmentation de voitures avec comparaison au masque réel.")

    # Chargement des fichiers
    uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])
    uploaded_mask = st.file_uploader("Choisissez un masque réel (optionnel)...", type=["png", "jpg", "jpeg"])

    # Sidebar avec légende et informations
    st.sidebar.title("Informations")
    st.sidebar.info("""
    Cette application utilise un modèle de deep learning pour segmenter 
    les images de scènes urbaines en 8 catégories différentes.
    """)

    # Si aucune image n'est chargée, afficher un message et s'arrêter là
    if uploaded_file is None:
        st.info("Veuillez télécharger une image pour commencer.")
        return

    # Charger et afficher l'image d'origine
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Image d'origine", use_container_width=True)

    # Créer la légende des classes
    legend_text, color_map = create_legend(CLASS_MAPPING_COLORS)
    st.sidebar.subheader("Légende des Objets")
    st.sidebar.markdown(legend_text, unsafe_allow_html=True)

    # === Section du masque réel ===
    real_mask_np = None
    colored_real_mask = None
    
    if uploaded_mask is not None:
        st.subheader("Masque Réel")
        try:
            real_mask = Image.open(uploaded_mask).convert('L')
            real_mask_np = np.array(real_mask)
            
            # Afficher le masque réel colorisé
            colored_real_mask = colorize_mask(real_mask_np, color_map)
            st.image(colored_real_mask, caption="Masque réel colorisé", use_container_width=True)
            
            # Superposition du masque réel sur l'image
            overlaid_real = overlay_mask(image, colored_real_mask, color_map)
            st.image(overlaid_real, caption="Superposition du masque réel", use_container_width=True)
        except Exception as e:
            st.error(f"Erreur lors du chargement du masque réel: {e}")

    # === Section de prédiction ===
    if st.button("Prédire le Mask"):
        with st.spinner("Prédiction en cours..."):
            # Obtenir le masque prédit de l'API
            success, result = predict_mask(image)
            
            if success:
                predicted_mask_img = result
                
                # Traiter le masque prédit
                predicted_mask_np = process_predicted_mask(predicted_mask_img)
                
                # Coloriser et afficher le masque prédit
                st.subheader("Masque Prédit")
                colored_predicted_mask = colorize_mask(predicted_mask_np, color_map)
                st.image(colored_predicted_mask, caption="Masque prédit colorisé", use_container_width=True)

                # Superposition du masque prédit sur l'image
                st.subheader("Superposition Prédite")
                overlaid_predicted = overlay_mask(image, colored_predicted_mask, color_map)
                st.image(overlaid_predicted, caption="Superposition du masque prédit", use_container_width=True)
                
                # Si un masque réel est disponible, faire une comparaison
                if real_mask_np is not None and colored_real_mask is not None:
                    st.subheader("Comparaison des Masques")
                    
                    # Comparer les masques
                    accuracy, resized_real_mask_np = compare_masks(real_mask_np, predicted_mask_np)
                    st.metric("Précision pixel par pixel", f"{accuracy:.2f}%")
                    
                    # Afficher les masques côte à côte pour comparaison visuelle
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(colored_real_mask, caption="Masque réel", use_container_width=True)
                    with col2:
                        st.image(colored_predicted_mask, caption="Masque prédit", use_container_width=True)
            else:
                # Afficher le message d'erreur retourné par la fonction predict_mask
                st.error(result)
                
                # Si c'est un objet Exception, afficher le stacktrace complet
                if isinstance(result, Exception):
                    st.exception(result)


if __name__ == "__main__":
    main()