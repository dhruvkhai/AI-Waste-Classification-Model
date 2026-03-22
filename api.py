import tensorflow as tf
import numpy as np
import io
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = FastAPI(title="Waste Classification API", description="API for classifying waste using MobileNetV2")

# Global variable to hold the model
MODEL = None
MODEL_PATH = "waste_classifier_final.h5"
CLASS_NAMES = ['biodegradable', 'hazardous', 'recyclable']

@app.on_event("startup")
async def load_model_on_startup():
    """ Load the model when the API starts up to have it ready for requests """
    global MODEL
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        try:
            MODEL = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Warning: Model file '{MODEL_PATH}' not found. Prediction endpoint will fail until it's available.")

@app.get("/")
def home():
    return {"message": "Welcome to the Waste Classification API. Go to /docs for the Swagger UI."}

@app.post("/predict")
async def predict_waste(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    try:
        # Read the image file bytes
        contents = await file.read()
        
        # Load the image using PIL
        img = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if it has an alpha channel or is grayscale
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize image to match model input (224, 224)
        img = img.resize((224, 224))
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Expand dimensions to create a batch of 1
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess input (very important, must match training)
        img_array = preprocess_input(img_array)
        
        # Make predicting
        predictions = MODEL.predict(img_array)
        
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        return JSONResponse(content={
            "success": True,
            "prediction": predicted_class,
            "confidence": confidence,
            "filename": file.filename
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# To run the API, use the command:
# uvicorn api:app --reload
