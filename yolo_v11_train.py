"""YOLOv11 training script for waste detection.
Assumes dataset in `DATASET/` following YOLO format.
"""
import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv11 model for waste detection")
    parser.add_argument("--data", type=str, default="DATASET/data.yaml", help="Path to dataset yaml file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--img", type=int, default=640, help="Image size")
    parser.add_argument("--project", type=str, default="runs/train", help="Project folder for runs")
    return parser.parse_args()

def main():
    args = parse_args()
    model = YOLO("yolo11n.pt")  # using nano version as base
    model.train(data=args.data, epochs=args.epochs, batch=args.batch, imgsz=args.img, project=args.project)

if __name__ == "__main__":
    main()
