from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import os
app = Flask(__name__)
from tensorflow.keras.models import Model, load_model



# Load your trained model
MODEL_PATH = "./models/chillimodel2.h5"  
model = load_model(MODEL_PATH,compile=False)



# Define classes
classes = ['cerocospora','healthy','murda complex','nutritional deficiency','powdery mildew']

# Helper function to preprocess image
def preprocess_image(image, target_size=(256, 256)):
    image = image.resize(target_size)  # Resize image to match model input
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/classify', methods=['POST'])
def classify():
    # Check if an image is uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    try:
        # Open the uploaded image
        image = Image.open(file.stream)
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make a prediction
        predictions = model.predict(processed_image)
        predicted_class = classes[np.argmax(predictions)]
        
        # Send response
        response = {
            'prediction': predicted_class,
            'confidence': float(np.max(predictions))
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return "Welcome to the Plant Disease Detection API of Chilli Model! Use the `/classify` endpoint to classify images."

if __name__ == '__main__':
    app.run(debug=True)
