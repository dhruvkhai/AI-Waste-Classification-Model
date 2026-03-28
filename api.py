import tensorflow as tf  # type: ignore
import numpy as np  # type: ignore
import io
import os
import cv2
import time
import shutil
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException  # type: ignore
from fastapi.responses import JSONResponse  # type: ignore
from PIL import Image  # type: ignore
from ultralytics import YOLO  # type: ignore
from mobile_classification_models import classify_crop  # type: ignore

app = FastAPI(title="Waste Classification & Detection API", description="API for detecting objects and classifying waste types")

# Configuration
DETECTION_MODEL = None
DETECTION_MODEL_PATH = "yolo11n.pt"
COLLECTION_DIR = "TESTED_DATASET"

@app.on_event("startup")
async def load_models_on_startup():
    """ Load both models when the API starts up """
    global DETECTION_MODEL
    
    # Load YOLO detection model
    if os.path.exists(DETECTION_MODEL_PATH):
        print(f"Loading detection model from {DETECTION_MODEL_PATH}...")
        try:
            DETECTION_MODEL = YOLO(DETECTION_MODEL_PATH)
            print("Detection model loaded successfully!")
        except Exception as e:
            print(f"Error loading detection model: {e}")
    else:
        print(f"Warning: Detection model file '{DETECTION_MODEL_PATH}' not found. Multi-object detection will fail.")

def save_to_dataset(img_bytes, filename, label):
    """
    Saves the uploaded image into the collection directory organized by label.
    """
    if not os.path.exists(COLLECTION_DIR):
        os.makedirs(COLLECTION_DIR)
        print(f"Created new collection directory: {COLLECTION_DIR}")
    
    label_dir = os.path.join(COLLECTION_DIR, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    
    # Generate unique filename with increasing suffix
    ext = os.path.splitext(filename)[1]
    if not ext: ext = ".jpg"
    existing_files = [f for f in os.listdir(label_dir) if f.startswith(label) and f.endswith(ext)]
    next_index = len(existing_files) + 1
    new_filename = f"{label}_{next_index}{ext}"
    target_path = os.path.join(label_dir, new_filename)
    
    with open(target_path, "wb") as f:
        f.write(img_bytes)
    return target_path

@app.get("/")
def home():
    return {"message": "Welcome to the Waste Classification & Detection API. Go to /docs for the Swagger UI."}

@app.post("/predict")
async def predict_waste(files: List[UploadFile] = File(...)):
    if DETECTION_MODEL is None:
        raise HTTPException(status_code=503, detail="Detection model is not loaded.")
        
    results_dict = {}
    
    for file in files:
        try:
            start_time = time.time()
            # Read the image file bytes
            contents = await file.read()
            
            # Convert to numpy array for OpenCV
            nparr = np.frombuffer(contents, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                results_dict[file.filename] = {"error": "Could not decode image."}
                continue
                
            # Run detection
            results = DETECTION_MODEL(img_bgr, verbose=False)
            detections = results[0]
            
            output_detections = []
            
            # Primary label for saving (usually the most confident waste category)
            top_waste_category = "unknown"
            top_waste_conf = 0.0

            if hasattr(detections, 'boxes') and detections.boxes is not None:
                for box in detections.boxes:
                    conf_val = float(box.conf.item())
                    if conf_val < 0.2: continue
                    
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, xyxy)
                    cls_idx = int(box.cls[0].item())
                    obj_name = DETECTION_MODEL.names[cls_idx]
                    
                    # Crop and Classify
                    h, w = img_bgr.shape[:2]
                    x1_c, y1_c = max(0, x1), max(0, y1)
                    x2_c, y2_c = min(w, x2), min(h, y2)
                    
                    crop = img_bgr[y1_c:y2_c, x1_c:x2_c]
                    if crop is None or crop.size == 0: continue
                    
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    waste_category, waste_conf = classify_crop(crop_rgb)
                    
                    if waste_conf > top_waste_conf:
                        top_waste_conf = waste_conf
                        top_waste_category = waste_category

                    output_detections.append({
                        "bbox": [x1_c, y1_c, x2_c, y2_c],
                        "object_name": str(obj_name),
                        "waste_category": str(waste_category),
                        "confidence": float(conf_val),
                        "waste_confidence": float(waste_conf)
                    })
            
            # Save to dataset
            if output_detections:
                save_to_dataset(contents, file.filename, top_waste_category)
            
            end_time = time.time()
            processing_time_ms = (end_time - start_time) * 1000
            
            results_dict[file.filename] = {
                "success": True,
                "objects_found": len(output_detections),
                "detections": output_detections,
                "processing_time_ms": round(processing_time_ms, 2)
            }
            
        except Exception as e:
            results_dict[file.filename] = {"success": False, "error": str(e)}
            
    return JSONResponse(content={
        "success": True,
        "results": results_dict,
        "saved_to": COLLECTION_DIR
    })

# To run the API, use the command:
# python -m uvicorn api:app --reload
