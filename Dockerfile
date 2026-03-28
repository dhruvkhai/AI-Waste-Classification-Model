# Use a lightweight python image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for OpenCV and tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files first to leverage Docker cache
COPY requirements-api.txt .
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --default-timeout=100 -r requirements-api.txt -r requirements.txt

# Copy application files
COPY api.py .
COPY predict.py .
COPY mobile_classification_models.py .
COPY inference.py .
COPY waste_classifier_final.h5 .
COPY yolo11n.pt .

# Create the collection directory
RUN mkdir -p TESTED_DATASET

# Expose port 8000 for the FastAPI server
EXPOSE 8000

# Command to run the FastAPI server using Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
