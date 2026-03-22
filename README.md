# ♻️ AI Waste Classification Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Modern-green.svg)
![MobileNetV2](https://img.shields.io/badge/Model-MobileNetV2-blueviolet)

An intelligent, lightweight, and blazing-fast image classification system that categorizes waste into **Biodegradable**, **Recyclable**, and **Hazardous** categories using a fine-tuned MobileNetV2 model.

Designed for edge deployment (e.g., smart bins or raspberry pi) with built-in support for real-time predictions via a robust **FastAPI backend**.

---

## ✨ Key Features
- **High Accuracy & Speed**: Uses `MobileNetV2` for rapid, real-time image inference.
- **Pre-trained Edge Model**: Includes a highly optimized `.h5` model with automatic conversion scripts to `TensorFlow Lite (.tflite)` for ultra-low latency edge devices.
- **Real-Time API**: A sleek `FastAPI` service that accepts image uploads and responds with the predicted waste class instantly.
- **Easy Pipeline**: Modular scripts for data loading, training, predicting, and deploying.

---

## 🛠️ Technology Stack
* **Deep Learning Framework**: TensorFlow Core & Keras
* **Base Architecture**: MobileNetV2 (Transfer Learning)
* **Backend API API Server**: FastAPI & Uvicorn
* **Image Processing**: Pillow & NumPy

---

## 📂 Project Structure

```text
AI_Waste_Project/
├── api.py                      # FastAPI server for the real-time API
├── train.py                    # Script to train and fine-tune the model
├── predict.py                  # CLI tool to run predictions on images
├── data_loader.py              # Handles Dataset loading/augmentation
├── utils.py                    # Utility functions for metrics & visualizations
├── best_model.h5               # Model checkpoints during training
├── waste_classifier_final.h5   # Fully trained and optimized classification model
├── requirements-api.txt        # FastAPI dependencies
└── DATASET/                    # (Folder containing images split by class)
```

---

## 🚀 Getting Started

### 1. Installation

Ensure you have Python 3.8+ installed. Clone this repository and install the dependencies for the API:

```bash
pip install -r requirements-api.txt
```

*(Note: If you plan to retrain the model, make sure you have the full required deep learning tools installed like `tensorflow`).*

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
docker build -t waste-classifier-api .
```

**Run the Docker container:**
```bash
docker run -d -p 8000:8000 --name waste-api waste-classifier-api
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

If you would like to retrain the model on your own newly gathered data:

1. Ensure your data is organized inside the `DATASET/` folder with sub-folders corresponding to each class (e.g., `biodegradable`, `recyclable`, `hazardous`).
2. Run the `train.py` script to retrain the `MobileNetV2` head:
   ```bash
   python train.py
   ```
3. Use the `predict.py` tools to test your model locally without the API:
   ```bash
   python predict.py --image path_to_image.jpg
   python predict.py --convert  # Converts model to .tflite format for Edge!
   ```

---

*Let's build a sustainable future with AI! 🌍*
