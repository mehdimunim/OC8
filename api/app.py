"""
API Flask pour la Segmentation d'Images de Voitures Autonomes

Cette API fournit un point d'accès pour la segmentation d'images urbaines.
Elle charge un modèle UNet entraîné et expose un endpoint REST qui:
1. Reçoit une image via une requête POST
2. Prétraite l'image
3. Effectue la segmentation (prédiction)
4. Renvoie le masque segmenté en format base64

Le modèle utilise des métriques et fonctions de perte personnalisées 
(Dice coefficient, IoU, mixed loss) optimisées pour la segmentation d'images.
"""

from flask import Flask, request, jsonify
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K
import base64

# ------------------------------------------------------------------------------
# Configuration et initialisation
# ------------------------------------------------------------------------------

app = Flask(__name__)

# Définir le chemin du modèle
MODEL_PATH = 'model/VGG16_UNet_dice_loss.keras'

# Variables globales pour les dimensions du modèle
IMG_HEIGHT = None
IMG_WIDTH = None
NUM_CLASSES = None
model = None

# ------------------------------------------------------------------------------
# Fonctions de perte et métriques personnalisées
# ------------------------------------------------------------------------------

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Calcule le coefficient de Dice entre les masques réel et prédit.
    
    Le coefficient de Dice mesure la similarité entre deux ensembles (ici masques binaires)
    et est défini comme 2 * |X ∩ Y| / (|X| + |Y|)
    
    Args:
        y_true (tensor): Masque réel
        y_pred (tensor): Masque prédit
        smooth (float): Petit epsilon pour éviter les divisions par zéro
    
    Returns:
        float: Coefficient de Dice entre 0 (pas de correspondance) et 1 (correspondance parfaite)
    """
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def iou_metric(y_true, y_pred, smooth=1e-6):
    """
    Calcule la métrique IoU (Intersection over Union) entre les masques réel et prédit.
    
    L'IoU est défini comme |X ∩ Y| / |X ∪ Y|
    
    Args:
        y_true (tensor): Masque réel
        y_pred (tensor): Masque prédit
        smooth (float): Petit epsilon pour éviter les divisions par zéro
    
    Returns:
        float: Score IoU entre 0 (pas de correspondance) et 1 (correspondance parfaite)
    """
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def dice_loss(y_true, y_pred):
    """
    Fonction de perte basée sur le coefficient de Dice.
    
    Args:
        y_true (tensor): Masque réel
        y_pred (tensor): Masque prédit
        
    Returns:
        float: Valeur de perte entre 0 et 1 (0 étant l'optimum)
    """
    return 1 - dice_coefficient(y_true, y_pred)


def mixed_loss(y_true, y_pred):
    """
    Fonction de perte combinant categorical crossentropy et dice loss.
    
    Cette fonction de perte composite équilibre les avantages de chaque métrique individuelle:
    - Categorical Crossentropy: bonne pour la classification pixel par pixel
    - Dice Loss: bonne pour gérer le déséquilibre de classes dans les masques de segmentation
    
    Args:
        y_true (tensor): Masque réel
        y_pred (tensor): Masque prédit
        
    Returns:
        float: Valeur de perte combinée
    """
    return 0.5 * keras.losses.CategoricalCrossentropy()(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)


# ------------------------------------------------------------------------------
# Fonctions de traitement d'image
# ------------------------------------------------------------------------------

def preprocess_image(image):
    """
    Prétraite l'image pour l'entrée du modèle.
    
    Les étapes de prétraitement incluent:
    - Redimensionnement à la taille attendue par le modèle
    - Normalisation des valeurs de pixels entre 0 et 1
    - Ajout d'une dimension batch
    
    Args:
        image (PIL.Image): Image d'entrée
        
    Returns:
        ndarray: Image prétraitée prête pour l'inférence
    """
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    print(f"Image prétraitée avec dimensions: {image.shape}")
    return image


def decode_prediction(prediction):
    """
    Décode la prédiction (probabilités) en un masque d'indices de classe.
    
    Args:
        prediction (ndarray): Prédiction brute du modèle (B, H, W, NUM_CLASSES)
        
    Returns:
        ndarray: Masque d'indices de classe (H, W)
    """
    print(f"Taille de la prédiction brute: {prediction.shape}")
    mask = np.argmax(prediction, axis=-1)
    print(f"Masque décodé avec dimensions: {mask.shape}")
    return mask[0]  # Retirer la dimension batch


def colorize_mask(mask):
    """
    Colorise le masque segmenté pour la visualisation.
    
    Chaque classe est colorée avec une couleur aléatoire distincte.
    
    Args:
        mask (ndarray): Masque d'indices de classe (H, W)
        
    Returns:
        PIL.Image: Masque colorisé
    """
    # Créer une palette de couleurs aléatoires
    colors = np.random.randint(0, 256, size=(NUM_CLASSES, 3), dtype=np.uint8)
    
    # Appliquer les couleurs au masque
    colored_mask = colors[mask]
    print(f"Masque colorisé avec dimensions: {colored_mask.shape}")
    
    return Image.fromarray(colored_mask)


# ------------------------------------------------------------------------------
# Initialisation du modèle
# ------------------------------------------------------------------------------

def load_model():
    """
    Charge le modèle de segmentation avec les objets personnalisés.
    
    Returns:
        bool: True si le chargement a réussi, False sinon
    """
    global model, IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES
    
    print(f"Tentative de chargement du modèle depuis: {MODEL_PATH}")
    try:
        model = keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                'dice_loss': dice_loss,
                'dice_coefficient': dice_coefficient,
                'iou_metric': iou_metric,
                'mixed_loss': mixed_loss
            }
        )
        
        # Extraire les dimensions du modèle
        IMG_HEIGHT = model.input_shape[1]
        IMG_WIDTH = model.input_shape[2]
        NUM_CLASSES = model.output_shape[-1]
        
        print(f"Modèle chargé avec succès.")
        print(f"Dimensions d'entrée: {IMG_HEIGHT}x{IMG_WIDTH}")
        print(f"Nombre de classes: {NUM_CLASSES}")
        
        return True
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return False


# Charger le modèle au démarrage
model_loaded = load_model()

# ------------------------------------------------------------------------------
# Routes de l'API
# ------------------------------------------------------------------------------

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint pour la prédiction de segmentation d'image.
    
    Reçoit une image via une requête POST, effectue la segmentation
    et renvoie le masque prédit encodé en base64.
    
    Returns:
        JSON: Réponse contenant le masque encodé ou un message d'erreur
    """
    print("Requête de prédiction reçue.")
    
    # Vérifier si le modèle est chargé
    if model is None:
        return jsonify({'error': 'Le modèle n\'a pas été chargé correctement.'}), 500

    # Vérifier la présence d'une image dans la requête
    if 'image' not in request.files:
        return jsonify({'error': 'Pas d\'image fournie dans la requête.'}), 400

    image_file = request.files['image']
    print(f"Nom du fichier reçu: {image_file.filename}")
    
    if image_file.filename == '':
        return jsonify({'error': 'Nom de fichier image vide.'}), 400

    try:
        # Traitement et prédiction
        print("Lecture et traitement de l'image...")
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        
        print("Prétraitement de l'image...")
        processed_image = preprocess_image(image)
        
        print("Prédiction en cours...")
        prediction = model.predict(processed_image)
        
        print("Décodage du masque...")
        decoded_mask = decode_prediction(prediction)
        
        print("Colorisation du masque...")
        colored_mask = colorize_mask(decoded_mask)

        # Conversion en base64 pour la réponse JSON
        print("Conversion du masque en base64...")
        img_io = io.BytesIO()
        colored_mask.save(img_io, 'PNG')
        img_io.seek(0)
        mask_bytes = img_io.getvalue()
        
        # Encodage en base64 pour une transmission plus sûre
        mask_base64 = base64.b64encode(mask_bytes).decode('utf-8')

        print("Réponse envoyée avec succès.")
        return jsonify({'mask': mask_base64})
        
    except Exception as e:
        print(f"Erreur lors du traitement de l'image ou de la prédiction: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint pour vérifier l'état de santé de l'API.
    
    Returns:
        JSON: Statut de l'API et du modèle
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_info': {
            'path': MODEL_PATH,
            'input_dimensions': f"{IMG_HEIGHT}x{IMG_WIDTH}" if IMG_HEIGHT else "Non disponible",
            'num_classes': NUM_CLASSES
        } if model is not None else None
    })


# ------------------------------------------------------------------------------
# Démarrage de l'application
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    print("Démarrage de l'API Flask...")
    app.run(debug=True, host='0.0.0.0')
    print("API arrêtée.")