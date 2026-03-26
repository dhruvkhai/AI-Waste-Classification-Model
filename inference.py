import argparse
import os
import cv2
import numpy as np
import json
from ultralytics import YOLO

# Import the classification helper
from mobile_classification_models import classify_crop

def load_yolo_model(weights_path='models/yolo_v11_best.pt'):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"YOLO weights not found at {weights_path}")
    model = YOLO(weights_path)
    return model

def run_detection(model, img_path, img_size=640):
    img_size_int = int(img_size)
    results = model(img_path, imgsz=img_size_int)
    return results[0]

def crop_and_classify(img_path, detections, model_names, conf_thresh=0.3):
    return crop_and_classify_tile(img_path, detections, model_names, x_offset=0, y_offset=0, conf_thresh=float(conf_thresh))

def crop_and_classify_tile(img_path, detections, model_names, x_offset=0, y_offset=0, conf_thresh=0.3):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")
    h, w = img.shape[:2]
    h_int, w_int = int(h), int(w)
    output = []
    
    seen_boxes = []
    
    def is_duplicate(new_box_list, existing_boxes_list, iou_thresh=0.5):
        for box in existing_boxes_list:
            nx1, ny1, nx2, ny2 = int(new_box_list[0]), int(new_box_list[1]), int(new_box_list[2]), int(new_box_list[3])
            bx1, by1, bx2, by2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            
            ix1, iy1 = max(nx1, bx1), max(ny1, by1)
            ix2, iy2 = min(nx2, bx2), min(ny2, by2)
            
            inter_area = max(0, int(ix2 - ix1)) * max(0, int(iy2 - iy1))
            box1_area = int(nx2 - nx1) * int(ny2 - ny1)
            box2_area = int(bx2 - bx1) * int(by2 - by1)
            union_area = box1_area + box2_area - inter_area
            
            if union_area > 0 and (float(inter_area) / float(union_area)) > float(iou_thresh):
                return True
        return False

    if not hasattr(detections, 'boxes') or detections.boxes is None:
        return []

    for box in detections.boxes:
        conf_val = float(box.conf.item())
        if conf_val < float(conf_thresh):
            continue
        
        xyxy_raw = box.xyxy[0].tolist()
        tx1 = int(float(xyxy_raw[0]))
        ty1 = int(float(xyxy_raw[1]))
        tx2 = int(float(xyxy_raw[2]))
        ty2 = int(float(xyxy_raw[3]))
        
        gx1 = int(tx1 + int(float(x_offset)))
        gy1 = int(ty1 + int(float(y_offset)))
        gx2 = int(tx2 + int(float(x_offset)))
        gy2 = int(ty2 + int(float(y_offset)))
        
        gx1_c, gy1_c = max(0, gx1), max(0, gy1)
        gx2_c, gy2_c = min(w_int, gx2), min(h_int, gy2)
        
        if is_duplicate([gx1_c, gy1_c, gx2_c, gy2_c], seen_boxes):
            continue
        seen_boxes.append([gx1_c, gy1_c, gx2_c, gy2_c])
        
        cls_raw = box.cls[0].item()
        class_id = int(float(cls_raw))
        yolo_class_name = str(model_names[class_id])
        
        crop = img[gy1_c:gy2_c, gx1_c:gx2_c]
        if crop is None or crop.size == 0: continue
            
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        class_name, class_conf = classify_crop(crop_rgb)
        
        output.append({
            "bbox": [int(gx1_c), int(gy1_c), int(gx2_c), int(gy2_c)],
            "detected_obj": str(yolo_class_name),
            "detection_confidence": float(conf_val),
            "classified_as": str(class_name),
            "classification_confidence": float(class_conf),
        })
    return output

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv11 detection and classify crops.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image file.")
    group.add_argument("--folder", type=str, help="Path to a folder of images.")
    parser.add_argument("--weights", type=str, default="runs/detect/train2/weights/best.pt", help="Custom YOLOv11 weights.")
    parser.add_argument("--fallback_weights", type=str, default="yolo11n.pt", help="Fallback YOLOv11 weights.")
    parser.add_argument("--conf", type=float, default=0.1, help="Detection confidence threshold.")
    args = parser.parse_args()

    custom_model = None
    if os.path.exists(args.weights):
        custom_model = load_yolo_model(args.weights)
    
    fallback_model = None
    if os.path.exists(args.fallback_weights):
        fallback_model = load_yolo_model(args.fallback_weights)

    if not custom_model and not fallback_model:
        print("Error: No valid YOLO models found.")
        return

    image_paths = []
    if args.image:
        image_paths = [args.image]
    elif args.folder and os.path.exists(args.folder):
        for f in os.listdir(args.folder):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(args.folder, f))

    all_results = {}
    for img_path in image_paths:
        try:
            final_classified = []
            img = cv2.imread(img_path)
            if img is None: continue
            h, w = img.shape[:2]
            h_int, w_int = int(h), int(w)
            
            if custom_model:
                for sz in [640, 1024]:
                    d = run_detection(custom_model, img_path, img_size=sz)
                    if d and hasattr(d, 'boxes') and d.boxes is not None:
                        results = crop_and_classify(img_path, d, custom_model.names, conf_thresh=args.conf)
                        for r in results:
                            if r not in final_classified:
                                final_classified.append(r)
            
            if len(final_classified) < 2 and fallback_model:
                t_size = 640
                overlap = 0.2
                stride = int(t_size * (1 - overlap))
                for y in range(0, h_int, stride):
                    for x in range(0, w_int, stride):
                        x_end = min(int(x + t_size), w_int)
                        y_end = min(int(y + t_size), h_int)
                        tile = img[int(y):int(y_end), int(x):int(x_end)]
                        if tile is None or tile.size == 0: continue
                        
                        f_res_list = fallback_model(tile, imgsz=640, verbose=False, conf=0.01)
                        if f_res_list and len(f_res_list) > 0:
                            res = f_res_list[0]
                            if res and hasattr(res, 'boxes') and res.boxes is not None:
                                WASTE_CLASSES = ['bottle', 'cup', 'wine glass', 'knife', 'fork', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'book', 'scissors', 'backpack', 'handbag', 'suitcase']
                                EXCLUDE_CLASSES = ['dining table', 'bed', 'sink', 'wall', 'floor', 'desk', 'person', 'chair', 'couch', 'potted plant', 'tv', 'laptop', 'mouse', 'keyboard', 'cell phone']
                                
                                tile_results = crop_and_classify_tile(img_path, res, fallback_model.names, x_offset=int(x), y_offset=int(y), conf_thresh=0.01)
                                for tr in tile_results:
                                    d_obj = str(tr.get("detected_obj", ""))
                                    if d_obj in EXCLUDE_CLASSES: continue
                                    try:
                                        c_conf = float(tr.get("classification_confidence", 0.0))
                                        if d_obj in WASTE_CLASSES or (c_conf > 0.95 and d_obj not in EXCLUDE_CLASSES):
                                            if tr not in final_classified:
                                                final_classified.append(tr)
                                    except (ValueError, TypeError): continue

            img_base = str(os.path.basename(img_path))
            all_results[img_base] = {
                "objects_found": int(len(final_classified)),
                "detections": list(final_classified),
                "status": str("success" if len(final_classified) > 0 else "no objects detected")
            }
        except Exception as e:
            all_results[str(os.path.basename(str(img_path)))] = {"status": "error", "message": str(e)}

    print("\n--- FINAL OUTPUT ---")
    print(json.dumps(all_results, indent=2))

if __name__ == "__main__":
    main()
