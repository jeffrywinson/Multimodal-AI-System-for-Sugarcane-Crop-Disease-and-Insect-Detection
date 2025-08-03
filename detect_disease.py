# detect_disease.py
from ultralytics import YOLO
import argparse
import torch

def predict_disease(image_path):
    # Load the trained YOLOv8s-seg model
    model_path = './YOLOv8s-seg/best.pt'
    model = YOLO(model_path)

    # Run prediction on the image
    results = model.predict(image_path, verbose=False)

    # Check if any masks (disease areas) were detected
    # results[0].masks will be None if no segmentation is found
    if results[0].masks is not None and len(results[0].masks) > 0:
        print("Present")
    else:
        print("Not Present")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect disease in an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    args = parser.parse_args()
    predict_disease(args.image)