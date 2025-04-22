from flask import Flask, request, jsonify
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K

app = Flask(__name__)

# Définir le chemin du modèle
model_path = 'model/UNet_base_mixed_loss_augmented.keras'

# Définir les fonctions de perte et les métriques personnalisées ICI
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def mixed_loss(y_true, y_pred):
    """Fonction de perte combinant categorical crossentropy et dice loss."""
    return 0.5 * keras.losses.CategoricalCrossentropy()(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)

# Charger le modèle en spécifiant les objets personnalisés
print("Tentative de chargement du modèle...")
try:
    model = keras.models.load_model(model_path,
                                   custom_objects={'dice_loss': dice_loss,
                                                   'dice_coefficient': dice_coefficient,
                                                   'iou_metric': iou_metric,
                                                   'mixed_loss': mixed_loss})
    print(f"Modèle chargé avec succès depuis : {model_path}")
    IMG_HEIGHT = model.input_shape[1]
    IMG_WIDTH = model.input_shape[2]
    NUM_CLASSES = model.output_shape[-1]
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    model = None
    IMG_HEIGHT = None
    IMG_WIDTH = None
    NUM_CLASSES = None

def preprocess_image(image):
    """Prétraite l'image pour l'entrée du modèle."""
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    print("Image prétraitée.")
    return image

def decode_prediction(prediction):
    """Décode la prédiction en un mask coloré."""
    print(f"Taille de la prédiction brute : {prediction.shape}")
    mask = np.argmax(prediction, axis=-1)
    print("Mask décodé.")
    return mask[0]

def colorize_mask(mask):
    """Colorise le mask segmenté pour la visualisation."""
    colors = np.random.randint(0, 256, size=(NUM_CLASSES, 3), dtype=np.uint8)
    colored_mask = colors[mask]
    print("Mask colorisé.")
    print(f"Taille du mask coloré : {colored_mask.size}")
    return Image.fromarray(colored_mask)

@app.route('/predict', methods=['POST'])
def predict():
    print("Requête de prédiction reçue.")
    if model is None:
        return jsonify({'error': 'Le modèle n\'a pas été chargé.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'Pas d\'image fournie'}), 400

    image_file = request.files['image']
    print(f"Nom du fichier reçu : {image_file.filename}")
    if image_file.filename == '':
        return jsonify({'error': 'Nom de fichier image vide'}), 400

    try:
        print("Prédiction du mask en cours...")
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        decoded_mask = decode_prediction(prediction)
        colored_mask = colorize_mask(decoded_mask)

        # Convertir le mask coloré en bytes pour l'envoyer dans la réponse
        img_io = io.BytesIO()
        colored_mask.save(img_io, 'PNG')
        img_io.seek(0)
        mask_bytes = img_io.getvalue()

        print("Mask converti en bytes et envoyé dans la réponse.")
        return jsonify({'mask': mask_bytes.decode('latin-1')}) # Encodage pour transmettre les bytes en JSON
    except Exception as e:
        print(f"Erreur lors du traitement de l'image ou de la prédiction : {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Démarrage de l'API Flask...")
    app.run(debug=True, host='0.0.0.0')
    print(f"L'API est en cours d'exécution sur http://{request.host_url}")