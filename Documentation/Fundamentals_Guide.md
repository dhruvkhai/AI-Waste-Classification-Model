# 📚 Project Fundamentals & Technical Concepts Guide

This document explains all the foundational technologies, frameworks, and concepts used in the AI Waste Classification System from the ground up. It is structured to help beginners understand *what* tools we used and *how* they work.

---

## 1. Image Classification & Deep Learning
**What is it?**
Image classification is a fundamental task in computer vision where a machine is trained to assign a label (or class) to an entire image. In this project, the labels are `Biodegradable`, `Recyclable`, and `Hazardous`.
**How it works:**
Instead of humans manually programming rules (e.g., "if it's green and shaped like a leaf, it's biodegradable"), Deep Learning models learn these patterns automatically by looking at thousands of examples during the **training phase**.

## 2. Convolutional Neural Networks (CNNs)
**What is it?**
A CNN is a specific type of Artificial Neural Network designed specifically to process pixel data and recognize shapes, textures, and objects in images.
**How it works:**
- **Convolution Layers**: These act as "filters" that slide over the image to detect features like edges, corners, and textures.
- **Pooling Layers**: These reduce the size of the image to make computations faster while keeping the most important information.
- **Dense Layers**: The final layers that piece all the detected features together to make the final prediction.

## 3. MobileNetV2
**What is it?**
MobileNetV2 is a highly efficient CNN architecture developed by Google. It is designed specifically to run on mobile devices and "edge" devices (like self-driving cars or smart trash bins) where computing power and battery life are limited.
**How it works:**
Traditional CNNs require massive amounts of mathematical operations. MobileNetV2 uses a technique called **Depthwise Separable Convolutions**. Instead of calculating a standard 3D filter across all color channels at once, it breaks the math down into two simpler, much faster steps. This drastically reduces the size of the model and the time it takes to process an image, without sacrificing much accuracy.

## 4. Transfer Learning
**What is it?**
Transfer Learning is a technique where we take a model that has already been trained on millions of generic images (like dogs, cars, and trees) and "transfer" its knowledge to our specific task (waste classification).
**How it works:**
Since MobileNetV2 has already learned how to see shapes, edges, and objects, we freeze its core "brain" and only delete and replace its final guessing layer. We then train this new final layer specifically on our garbage photos. This allows us to achieve incredibly high accuracy with very little training time and a relatively small dataset.

## 5. TensorFlow and Keras
**What are they?**
- **TensorFlow** is an open-source machine learning platform developed by Google.
- **Keras** is a high-level Python library that runs on top of TensorFlow, making it easier for developers to write deep learning code simply and intuitively.
**How we use them:**
We use Keras to load the MobileNetV2 blueprint, attach our custom layers, load our dataset, and execute the complex calculus required to train the model over several "epochs" (training rounds).

## 6. FastAPI (The Backend API)
**What is it?**
FastAPI is a modern, blazing-fast web framework for building Application Programming Interfaces (APIs) in Python.
**How it works:**
An API is a messenger that takes requests and tells a system what you want to do. Our FastAPI server sits and listens for HTTP requests. When a user or a smart camera sends an image to the API's `/predict` endpoint, FastAPI routes the image to our loaded TensorFlow model, interprets the prediction, and replies instantly with a clean JSON format.

## 7. Docker
**What is it?**
Docker is a tool designed to make it easier to create, deploy, and run applications by using "containers."
**How it works:**
A container allows us to package up an application (our FastAPI server and the AI model) with all of its exact dependencies, libraries, and Python versions into a single box. This means that our AI will run seamlessly on any computer, server, or cloud platform in the world, exactly the same way it runs on our personal laptop, without "it works on my machine" bugs.
