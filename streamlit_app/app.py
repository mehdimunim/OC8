import streamlit as st
import requests
from PIL import Image
import numpy as np
import io
import os
import cv2

# Titre et sous-titre de l'application
st.title("Application de Segmentation d'Images de Voitures")
st.subheader("Visualisation de la segmentation de voitures avec comparaison au masque réel.")

# --- Configuration ---
# Nombre de classes prédites par le modèle
NUM_CLASSES = 8

# Chemin vers le dossier des masques réels
REAL_MASK_DIR = "/home/mehdi/Documents/OC/OC8/data/processed/train/aachen"

# Mapping des classes aux noms d'objets et aux couleurs pour la légende et la colorisation
CLASS_MAPPING_COLORS = {
    0: ("void", [0, 0, 0]),         # Noir
    1: ("flat", [255, 255, 0]),        # Jaune
    2: ("human", [255, 0, 255]),        # Magenta
    3: ("vehicle", [0, 0, 255]),      # Bleu
    4: ("construction", [0, 255, 0]),   # Vert
    5: ("object", [255, 0, 0]),       # Rouge
    6: ("nature", [0, 255, 255]),      # Cyan
    7: ("sky", [255, 255, 255])         # Blanc
}

# --- Fonctions Utilitaires ---
# Fonction pour créer une légende de couleurs avec les noms des objets
def create_legend(class_mapping_colors):
    legend_items = []
    color_map = {}
    for class_index, (label, color) in class_mapping_colors.items():
        color_hex = '#%02x%02x%02x' % tuple(color)
        legend_items.append(f'<span style="background-color:{color_hex}; color:{color_hex}; padding: 5px; border-radius: 3px;">&nbsp;&nbsp;&nbsp;</span> {label}')
        color_map[class_index] = color
    return "<br>".join(legend_items), color_map

# Fonction pour coloriser un mask (prend un mask en niveaux de gris) avec une palette spécifique
def colorize_mask(mask, color_map):
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_index, color in color_map.items():
        mask_indices = (mask == class_index)
        if np.any(mask_indices):  # Vérifie si la classe existe dans le masque
            colored_mask[mask_indices] = color
    return Image.fromarray(colored_mask)

# Fonction pour superposer le masque sur l'image
def overlay_mask(image, mask, colors, alpha=0.5):
    img = np.array(image, dtype=np.uint8)
    mask_img = np.array(mask.convert("RGB"), dtype=np.uint8)
    masked_image = cv2.addWeighted(img, 1 - alpha, mask_img, alpha, 0)
    return Image.fromarray(masked_image)

# --- Interface Utilisateur Streamlit ---
# Upload de l'image par l'utilisateur
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Charger et afficher l'image d'origine
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Image d'origine", use_column_width=True)

    # Construire le chemin du masque réel correspondant
    image_name = os.path.splitext(uploaded_file.name)[0].replace("_image", "")
    real_mask_file_path = os.path.join(REAL_MASK_DIR, f"{image_name}_mask.png")

    # Créer la légende
    legend_text, color_map = create_legend(CLASS_MAPPING_COLORS)

    # --- Affichage du Masque Réel ---
    st.subheader("Masque Réel")
    if os.path.exists(real_mask_file_path):
        try:
            real_mask_np = cv2.imread(real_mask_file_path, cv2.IMREAD_UNCHANGED)
            if real_mask_np is not None:
                colored_real_mask = colorize_mask(real_mask_np, color_map)
                st.image(colored_real_mask, use_column_width=True)
            else:
                st.warning(f"Impossible de lire le fichier de masque réel: '{real_mask_file_path}'.")
        except FileNotFoundError:
            st.warning(f"Le fichier de masque réel '{real_mask_file_path}' n'a pas été trouvé.")
        except Exception as e:
            st.error(f"Erreur lors du chargement du masque réel: {e}")
    else:
        st.info(f"Le fichier de masque réel '{real_mask_file_path}' n'a pas été trouvé dans '{REAL_MASK_DIR}'.")

    # --- Prédiction et Affichage du Masque Prédit ---
    if st.button("Prédire le Mask"):
        files = {'image': uploaded_file.getvalue()}
        try:
            response = requests.post("http://localhost:5000/predict", files=files)
            response.raise_for_status()
            data = response.json()
            mask_bytes = data['mask'].encode('latin-1')
            predicted_mask = Image.open(io.BytesIO(mask_bytes))
            predicted_mask_np = np.array(predicted_mask)

            # Correction potentielle : s'assurer que le masque prédit est interprété comme des indices
            if predicted_mask_np.ndim == 3 and predicted_mask_np.shape[2] > 1:
                predicted_mask_np = np.argmax(predicted_mask_np, axis=-1)
                st.info("Le masque prédit semblait être une probabilité, converti en indices de classe.")

            colored_predicted_mask = colorize_mask(predicted_mask_np, color_map)
            st.subheader("Masque Prédit")
            st.image(colored_predicted_mask, use_column_width=True)

            # --- Affichage de la Superposition ---
            st.subheader("Superposition Prédite")
            overlaid_predicted = overlay_mask(image, colored_predicted_mask, color_map)
            st.image(overlaid_predicted, use_column_width=True)

            # --- Affichage de la Légende ---
            st.subheader("Légende des Objets")
            st.markdown(legend_text, unsafe_allow_html=True)

        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de la communication avec l'API: {e}")
        except KeyError:
            st.error("La réponse de l'API ne contient pas le champ 'mask'.")
        except Exception as e:
            st.error(f"Une erreur inattendue s'est produite: {e}")