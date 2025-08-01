import numpy as np
from PIL import Image

def preprocess_image(image, target_size=(256, 256)):
    image = image.resize(target_size)  # Resize image to match model input
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image