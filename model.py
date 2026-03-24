import tensorflow as tf  # type: ignore
from tensorflow.keras.applications import MobileNetV2  # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout  # type: ignore
from tensorflow.keras.models import Model  # type: ignore

def build_model(num_classes=3, input_shape=(224, 224, 3)):
    """
    Builds the MobileNetV2 based model for classification.
    """
    # Load the base MobileNetV2 model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the base model to prevent weights from updating during initial training
    base_model.trainable = False

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Adding a dense layer with dropout for regularization
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Final classification layer
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create the complete model
    model = Model(inputs=base_model.input, outputs=predictions)

    return model, base_model

def unfreeze_model(base_model, num_layers_to_unfreeze=30):
    """
    Unfreezes the top `num_layers_to_unfreeze` layers of the base model for fine-tuning.
    """
    base_model.trainable = True
    
    # Freeze all layers except the top 'num_layers_to_unfreeze'
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
        
    return base_model
