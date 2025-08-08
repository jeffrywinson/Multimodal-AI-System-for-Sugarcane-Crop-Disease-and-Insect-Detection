# detect_insect.py
from ultralytics import YOLO
import argparse
import torch

def predict_insect(image_path):
    # Load the trained YOLOv8s insect detection model
    model_path = './YOLOv8s/yolov8s_insect_detection_best.pt'
    model = YOLO(model_path)

    # Run prediction on the image
    results = model.predict(image_path, verbose=False)

    # Check if any bounding boxes (insects) were detected
    # Output the number of detected instances
    if results[0].boxes is not None:
        print(len(results[0].boxes))
    else:
        print(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect insects in an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    args = parser.parse_args()
    predict_insect(args.image)