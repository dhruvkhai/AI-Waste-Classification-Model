import os
import tensorflow as tf  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # type: ignore

def get_data_generators(dataset_dir, target_size=(224, 224), batch_size=32):
    """
    Creates training and validation data generators with data augmentation.
    Uses MobileNetV2's specific preprocess_input.
    """
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"Ensure that {train_dir} and {val_dir} exist.")

    # Data augmentation configuration for training
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Validation configuration (only preprocessing, no augmentation)
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    # Load and iterate training dataset
    print("Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    # Load and iterate validation dataset
    print("Loading validation data...")
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator
