import os
import tensorflow as tf  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # type: ignore
import numpy as np  # type: ignore

from data_loader import get_data_generators  # type: ignore
from model import build_model, unfreeze_model  # type: ignore
from utils import plot_training_history, plot_confusion_matrix, print_class_indices  # type: ignore

def train_pipeline(dataset_dir="dataset", epochs_initial=10, epochs_finetune=10, batch_size=32):
    """
    Complete training pipeline including data loading, transfer learning, fine-tuning, and evaluation.
    """
    target_size = (224, 224)
    num_classes = 3 # biodegradable, recyclable, hazardous
    
    # 1. Load Data
    train_generator, val_generator = get_data_generators(dataset_dir, target_size, batch_size)
    class_indices = train_generator.class_indices
    
    print("Class Indices Mapping:")
    for class_name, idx in class_indices.items():
        print(f" - {class_name}: {idx}")
    
    # 2. Build Model
    print("Building model...")
    model, base_model = build_model(num_classes=num_classes, input_shape=(224, 224, 3))
    
    # Initial compilation: train only the top layers with a standard learning rate
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    ]
    
    # 3. Train Initial Model (Transfer Learning)
    print("Starting initial training (training top layers only)...")
    history_initial = model.fit(
        train_generator,
        epochs=epochs_initial,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    # 4. Fine-Tuning
    print("Unfreezing base model top layers for fine-tuning...")
    base_model = unfreeze_model(base_model, num_layers_to_unfreeze=30)
    
    # Recompile with a much lower learning rate for fine-tuning
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("Starting fine-tuning...")
    # Fine-tune the model, picking up from where initial training left off
    history_finetune = model.fit(
        train_generator,
        epochs=epochs_initial + epochs_finetune,
        initial_epoch=history_initial.epoch[-1],
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    # Merge histories for plotting
    for key in history_initial.history.keys():
        history_initial.history[key].extend(history_finetune.history[key])
        
    class _MergedHistory:
        pass
    merged_history = _MergedHistory()
    merged_history.history = history_initial.history  # type: ignore
    
    # 5. Evaluation & Plotting
    print("Evaluating and plotting results...")
    plot_training_history(merged_history, save_path="training_history.png")
    
    # Confusion Matrix
    print("Generating predictions for confusion matrix...")
    val_generator.reset() # Reset generator before prediction
    predictions = model.predict(val_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes
    
    # Map back to class names
    class_names = list(class_indices.keys())
    # Sort class names based on index
    class_names = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]
    
    plot_confusion_matrix(y_true, y_pred, class_names, save_path="confusion_matrix.png")
    
    # Final Model Save
    model.save('waste_classifier_final.h5')
    print("Training complete. Final model saved as 'waste_classifier_final.h5'")

if __name__ == "__main__":
    import argparse
    import os

    # Automatically resolve the expected dataset path based on script location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(current_dir) == 'scripts':
        default_dataset = os.path.join(current_dir, '..', 'DATASET', 'dataset')
    else:
        default_dataset = os.path.join(current_dir, 'DATASET', 'dataset')

    parser = argparse.ArgumentParser(description="Train waste classification model.")
    parser.add_argument('--dataset', type=str, default=os.path.abspath(default_dataset), help="Path to the dataset directory")
    args = parser.parse_args()
    
    dataset_path = args.dataset
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' not found.")
        print("Use the --dataset argument to specify where the 'train' and 'val' folders are located.")
        print('Example: "C:\\Program Files\\Python\\python.exe" train.py --dataset "C:\\Path\\To\\Your\\dataset"')
    else:
        train_pipeline(dataset_dir=dataset_path)
