# Use a lightweight python image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# (libgl1-mesa-glx and libglib2.0-0 are often needed for cv2/Pillow but Pillow usually works without them. 
# We'll just install standard tools just in case, but keep it minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements-api.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy the rest of the application files
# We ignore unnecessary files via .dockerignore
COPY api.py .
COPY inference.py .
COPY waste_classifier_final.h5 .
COPY yolo11n.pt .

# Expose port 8000 for the FastAPI server
EXPOSE 8000

# Command to run the FastAPI server using Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
