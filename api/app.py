import os
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Charger le modèle
model_path = os.path.join(os.path.dirname(__file__), 'model', 'model.h5')
model = tf.keras.models.load_model(model_path, compile=False)

def preprocess_image(image):
    # Redimensionner l'image à la taille attendue par le modèle
    
    
    print(image)
    
    image = cv2.resize(image, (256, 256))
    # Normaliser l'image
    image = image.astype(np.float32) / 255.0
    return image

def postprocess_mask(mask):
    # Convertir le masque en image 8-bit
    mask = (mask * 255).astype(np.uint8)
    return mask

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    preprocessed_image = preprocess_image(image)
    
    # Faire la prédiction
    prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))[0]
    
    mask = postprocess_mask(prediction)
    
    # Encoder le masque en base64
    _, buffer = cv2.imencode('.png', mask)
    mask_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({'mask': mask_base64})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

