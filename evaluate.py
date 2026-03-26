import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import get_data_generators
from mobile_classification_models import MODEL_PATH, CLASS_NAMES, TARGET_SIZE

def evaluate_model(dataset_dir="DATASET/dataset", batch_size=32):
    """
    Evaluates the MobileNetV2 model on the validation set and prints detailed metrics.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Load data (only val_generator is needed)
    _, val_generator = get_data_generators(dataset_dir, target_size=TARGET_SIZE, batch_size=batch_size)
    
    print("Generating predictions...")
    val_generator.reset()
    predictions = model.predict(val_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes
    
    # Get class labels from generator to ensure correct mapping
    class_indices = val_generator.class_indices
    # Sort class names by their index value
    target_names = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]
    
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)
    
    # Also calculate overall accuracy explicitly
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    
    return report, accuracy

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate the waste classification model.")
    parser.add_argument('--dataset', type=str, default="DATASET/dataset", help="Path to the dataset directory containing 'val' folder")
    args = parser.parse_args()
    
    evaluate_model(dataset_dir=args.dataset)
