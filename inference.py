# inference.py
"""YOLOv11 inference script that runs detection on images and classifies each detected object using the MobileNetV2 classifier.

Usage:
    python inference.py --image path/to/image.jpg
    python inference.py --folder path/to/images_folder
"""
import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO

# Import the classification helper
from mobile_net_classifier import classify_crop

def load_yolo_model(weights_path='models/yolo_v11_best.pt'):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"YOLO weights not found at {weights_path}")
    model = YOLO(weights_path)
    return model

def run_detection(model, img_path, img_size=640):
    # YOLOv11 expects BGR images; cv2 reads as BGR
    results = model(img_path, imgsz=img_size)
    # results[0] contains boxes, scores, class ids
    return results[0]

def crop_and_classify(img_path, detections, conf_thresh=0.3):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    output = []
    for box in detections.boxes:
        conf = box.conf.item()
        if conf < conf_thresh:
            continue
        # box.xyxy returns (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        # Clip coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = img[y1:y2, x1:x2]
        # Convert crop to RGB for classifier
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        class_name, confidence = classify_crop(crop_rgb)
        output.append({
            "bbox": [x1, y1, x2, y2],
            "detection_confidence": conf,
            "class": class_name,
            "class_confidence": confidence,
        })
    return output

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv11 detection and classify crops.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image file.")
    group.add_argument("--folder", type=str, help="Path to a folder of images.")
    parser.add_argument("--weights", type=str, default="models/yolo_v11_best.pt", help="YOLOv11 checkpoint.")
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold.")
    args = parser.parse_args()

    model = load_yolo_model(args.weights)
    image_paths = []
    if args.image:
        image_paths = [args.image]
    else:
        for f in os.listdir(args.folder):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(args.folder, f))

    all_results = {}
    for img_path in image_paths:
        detections = run_detection(model, img_path)
        classified = crop_and_classify(img_path, detections, conf_thresh=args.conf)
        all_results[os.path.basename(img_path)] = classified
        print(f"Processed {img_path}: {len(classified)} objects classified.")
    # Output JSON summary
    import json
    print(json.dumps(all_results, indent=2))

if __name__ == "__main__":
    main()
