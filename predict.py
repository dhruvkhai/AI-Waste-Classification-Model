import tensorflow as tf  # type: ignore
import numpy as np  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # type: ignore
import os
import time
import shutil

# Configuration for data collection
COLLECTION_DIR = "TESTED_DATASET"

def load_trained_model(model_path='waste_classifier_final.h5'):
    """
    Loads the trained Keras model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Please train the model first.")
    
    print(f"Loading model from {model_path}...")
    return tf.keras.models.load_model(model_path)

def save_to_dataset(img_path, label):
    """
    Saves the tested image into the collection directory organized by label.
    """
    if not os.path.exists(COLLECTION_DIR):
        os.makedirs(COLLECTION_DIR)
        print(f"Created new collection directory: {COLLECTION_DIR}")
    
    label_dir = os.path.join(COLLECTION_DIR, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    
    # Generate unique filename with increasing suffix
    ext = os.path.splitext(img_path)[1]
    existing_files = [f for f in os.listdir(label_dir) if f.startswith(label) and f.endswith(ext)]
    next_index = len(existing_files) + 1
    new_filename = f"{label}_{next_index}{ext}"
    target_path = os.path.join(label_dir, new_filename)
    
    shutil.copy2(img_path, target_path)
    return target_path

def predict_image(model, img_path, class_names=['biodegradable', 'hazardous', 'recyclable']):
    """
    Loads an image, preprocesses it, and makes a prediction.
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image {img_path} not found.")

    start_time = time.time()
    
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    # Expand dimensions to match batch size format (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0) 
    
    # Preprocess input strictly same as training
    img_array = preprocess_input(img_array)

    # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    
    end_time = time.time()
    processing_time = (end_time - start_time) * 1000 # in ms
    
    predicted_class = class_names[predicted_class_idx]
    
    # Save to dataset
    collection_path = save_to_dataset(img_path, predicted_class)
    
    print(f"[{os.path.basename(img_path)}] Prediction: {predicted_class} (Conf: {confidence*100:.2f}%) | Time: {processing_time:.2f}ms")
    print(f"Saved to: {collection_path}")
    
    return predicted_class, confidence, processing_time

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
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"TFLite model saved to {tflite_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict waste class or convert model to TFLite")
    parser.add_argument('--image', type=str, help="Path to the image to classify")
    parser.add_argument('--folder', type=str, help="Path to a folder of images to classify")
    parser.add_argument('--convert', action='store_true', help="Convert the .h5 model to .tflite")
    parser.add_argument('--model', type=str, default='waste_classifier_final.h5', help="Path to the trained model")
    
    args = parser.parse_args()
    
    class_names_mapping = ['biodegradable', 'hazardous', 'recyclable']

    if args.convert:
        convert_to_tflite(model_path=args.model)
    elif args.image:
        model = load_trained_model(args.model)
        predict_image(model, args.image, class_names=class_names_mapping)
    elif args.folder:
        model = load_trained_model(args.model)
        if not os.path.isdir(args.folder):
            print(f"Error: {args.folder} is not a directory.")
        else:
            files = [f for f in os.listdir(args.folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            print(f"Processing {len(files)} images from {args.folder}...")
            total_start = time.time()
            for f in files:
                predict_image(model, os.path.join(args.folder, f), class_names=class_names_mapping)
            total_end = time.time()
            print(f"\n--- Summary ---")
            print(f"Total time for {len(files)} images: {total_end - total_start:.2f}s")
    else:
        print("Please specify an action. Use --help for options.")
