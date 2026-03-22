import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

def load_trained_model(model_path='waste_classifier_final.h5'):
    """
    Loads the trained Keras model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Please train the model first.")
    
    print(f"Loading model from {model_path}...")
    return tf.keras.models.load_model(model_path)

def predict_image(model, img_path, class_names=['biodegradable', 'hazardous', 'recyclable']):
    """
    Loads an image, preprocesses it, and makes a prediction.
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image {img_path} not found.")

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    # Expand dimensions to match batch size format (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0) 
    
    # Preprocess input strictly same as training
    img_array = preprocess_input(img_array)

    # Predict
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    
    predicted_class = class_names[predicted_class_idx]
    
    print(f"Prediction: {predicted_class} (Confidence: {confidence*100:.2f}%)")
    
    return predicted_class, confidence

def convert_to_tflite(model_path='waste_classifier_final.h5', tflite_path='waste_classifier.tflite'):
    """
    Converts a saved Keras `.h5` model to TensorFlow Lite format.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")

    print(f"Converting {model_path} to TFLite format...")
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Convert
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optional: Apply quantization here to make model smaller and faster
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()

    # Save
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"TFLite model saved to {tflite_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict waste class or convert model to TFLite")
    parser.add_argument('--image', type=str, help="Path to the image to classify")
    parser.add_argument('--convert', action='store_true', help="Convert the .h5 model to .tflite")
    parser.add_argument('--model', type=str, default='waste_classifier_final.h5', help="Path to the trained model")
    
    args = parser.parse_args()
    
    # Note: ensure class names list aligns with generator.class_indices
    # typically ascending alphabetical order when using flow_from_directory:
    class_names_mapping = ['biodegradable', 'hazardous', 'recyclable']

    if args.convert:
        convert_to_tflite(model_path=args.model)
    elif args.image:
        model = load_trained_model(args.model)
        predict_image(model, args.image, class_names=class_names_mapping)
    else:
        print("Please specify an action. Use --help for options.")
