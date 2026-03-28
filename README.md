# ♻️ AI Waste Classification Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Modern-green.svg)
![MobileNetV2](https://img.shields.io/badge/Model-MobileNetV2-blueviolet)

An intelligent, lightweight image pipeline that locates waste objects using **YOLOv11** and categorizes them into **Biodegradable**, **Recyclable**, and **Hazardous** categories using a fine-tuned **MobileNetV2** model.

Designed for edge deployment (e.g., smart bins or raspberry pi) with built-in support for real-time predictions via a robust **FastAPI backend**.

---

## ✨ Key Features

- **High Accuracy Object Detection & Classification**: Uses `YOLOv11` for precise object localization and `MobileNetV2` for rapid crop classification.
- **Pre-trained Edge Model**: Includes a highly optimized `.h5` model with automatic conversion scripts to `TensorFlow Lite (.tflite)` for ultra-low latency edge devices.
- **Real-Time API**: A sleek `FastAPI` service that accepts image uploads and responds with the predicted waste class instantly.
- **Easy Pipeline**: Modular scripts for data loading, training, predicting, and deploying.

---

## 🚀 What's New in the Latest YOLOv11 Model

- **Successfully Fully Trained**: Completed a full 50-epoch training run (`runs/detect/train2`) with excellent accuracy metrics.
- **Specific Object Names**: The updated model now outputs the specific object entity name when it detects waste.
- **28 Detailed Classes**: Training was fundamentally performed on 28 granular classes, seamlessly mapped into our core top-level categories.

---

## 🛠️ Technology Stack

- **Deep Learning Frameworks**: Ultralytics (YOLO) & TensorFlow Core / Keras (MobileNet)
- **Base Architectures**: YOLOv11 (Detection) & MobileNetV2 (Classification)
- **Backend API API Server**: FastAPI & Uvicorn
- **Image Processing**: OpenCV, Pillow & NumPy

---

## 📂 Project Structure

```text
AI_Waste_Project/
├── api.py                      # FastAPI server for the real-time API (MobileNet classification)
├── train.py                    # Script to train and fine-tune the MobileNetV2 model
├── predict.py                  # CLI tool to run predictions on images (MobileNet)
├── inference.py                # Pipeline script using YOLOv11 for detection, MobileNet for classification
├── yolo_v11_train.py           # Script to train the YOLOv11 model
├── data_loader.py              # Handles Dataset loading/augmentation for MobileNet
├── utils.py                    # Utility functions for metrics & visualizations
├── best_model.h5               # Model checkpoints during training
├── waste_classifier_final.h5   # Fully trained and optimized classification model
├── yolo11n.pt                  # YOLOv11 model weights
├── requirements-api.txt        # FastAPI dependencies
├── cluttered_waste_DATASET/    # YOLO detection dataset
└── DATASET/                    # MobileNet classification dataset
```

---

## 🚀 Getting Started

### 1. Installation

Ensure you have Python 3.8+ installed. Clone this repository and install the dependencies for the API:

```bash
pip install -r requirements-api.txt
```

_(Note: If you plan to retrain the model, make sure you have the full required deep learning tools installed like `tensorflow`)._

### 2. Running the API Server (Locally)

The project includes a blazing-fast REST API for real-time waste classification! Start the server running locally on your machine from your terminal:

```bash
uvicorn api:app --reload
```

The API will instantly start up and host a Swagger UI at: **[`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs)**. You can navigate here to test the model by uploading a photo directly from your web browser!

### 3. Running with Docker 🐳 (Recommended)

You can easily containerize and run the API anywhere using Docker, without needing to install Python or TensorFlow on your host machine!

**Build the Docker image:**

```bash
docker build -t waste_ai_deployed .
```

**Run the Docker container:**

```bash
docker run -d -p 8000:8000 --name waste-api_deployed waste_ai_deployed
```

Your API will now be live on `http://127.0.0.1:8000/docs` just like the local method, but completely isolated! You can stop the container anytime using `docker stop waste-api`.

---

## 📸 API Reference

### `POST /predict`

Upload an image of waste through this endpoint, and get an immediate JSON response with its predicted category!

**Example Curl Request:**

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@YOUR_IMAGE_PATH.jpg'
```

**Example JSON Response:**

```json
{
  "success": true,
  "prediction": "recyclable",
  "confidence": 0.985,
  "filename": "water_bottle.jpg"
}
```

---

## 🧠 Model Training (Optional)

If you would like to retrain the models on your own newly gathered data:

1. Ensure your classification data is organized inside the `DATASET/` folder with sub-folders corresponding to each class (e.g., `biodegradable`). For YOLO, use `cluttered_waste_DATASET/`.
2. Run the `train.py` script to retrain the `MobileNetV2` head:
   ```bash
   python train.py
   ```
3. Use the `predict.py` tools to test your model locally without the API:
   ```bash
   python predict.py --image path_to_image.jpg
   python predict.py --convert  # Converts model to .tflite format for Edge!
   ```

## Project Structure

- `api.py`: FastAPI server with multi-object detection and classification.
- `predict.py`: CLI tool for single image and folder-based classification.
- `mobile_classification_models.py`: Helper for MobileNetV2 classification.
- `inference.py`: YOLOv11 inference logic.
- `waste_classifier_final.h5`: Trained MobileNetV2 model.
- `yolo11n.pt`: Pre-trained YOLOv11 model.
- `TESTED_DATASET/`: Automatically created directory for collected test data.

## Testing Pipeline

### 1. Terminal (Bulk Folder Testing)
You can now test an entire folder of images at once:
```bash
python predict.py --folder "C:\path\to\your\images"
```
Each prediction will show the process time and will be saved to `TESTED_DATASET/`.

### 2. Web API (Multi-Image Testing)
The API at `/predict` now accepts multiple files.
- Go to `http://localhost:8000/docs`.
- Use the `/predict` POST endpoint.
- Select multiple images using the "Add string item" or by selecting multiple files in the browser dialog.
- The response will include timing for each image and the overall status.

## Data Collection
Every image tested through `predict.py` or the API is automatically saved in the `TESTED_DATASET/` folder, organized by the predicted label (`biodegradable`, `hazardous`, `recyclable`). This allows for continuous dataset expansion and model refinement.

---

_Let's build a sustainable future with AI! 🌍_