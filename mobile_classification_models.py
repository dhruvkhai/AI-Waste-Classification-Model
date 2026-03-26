import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

# Configuration
MODEL_PATH = "waste_classifier_final.h5"
CLASS_NAMES = ['biodegradable', 'hazardous', 'recyclable']
TARGET_SIZE = (224, 224)

# Global model variable for persistence
_MODEL = None

def get_model():
    global _MODEL
    if _MODEL is None:
        # Determine the correct model path (absolute path to avoid directory issues)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, MODEL_PATH)
        fallback_path = os.path.join(base_dir, "best_model.h5")
        
        if not os.path.exists(path):
            if os.path.exists(fallback_path):
                print(f"Loading from {fallback_path} as {MODEL_PATH} not found.")
                path = fallback_path
            else:
                raise FileNotFoundError(f"Neither {path} nor {fallback_path} found.")
        
        print(f"Loading MobileNetV2 model from {path}...")
        _MODEL = tf.keras.models.load_model(path)
    return _MODEL

def classify_crop(image_rgb):
    """
    Classifies a single image crop (numpy array in RGB format).
    Returns (class_name, confidence).
    """
    try:
        model = get_model()
        
        # Ensure image is in the correct format
        if not isinstance(image_rgb, np.ndarray):
            image_rgb = np.array(image_rgb)
        
        # Resize image to match model input
        img_resized = cv2.resize(image_rgb, TARGET_SIZE)
        
        # Prepare image for model
        img_array = np.expand_dims(img_resized, axis=0)
        img_array = preprocess_input(img_array.astype(np.float32))
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))
        
        return CLASS_NAMES[predicted_class_idx], confidence
    except Exception as e:
        print(f"Error during classification: {e}")
        return "unknown", 0.0
