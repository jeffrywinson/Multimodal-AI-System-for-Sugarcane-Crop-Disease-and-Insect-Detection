from ultralytics import YOLO
import argparse
import torch
import numpy as np

def predict_disease_area(image_path):
    # Load the trained YOLOv8s-seg model
    model_path = './YOLOv8s-seg/best.pt'
    model = YOLO(model_path)
    
    # Run prediction on the image
    results = model.predict(image_path, verbose=False)
    
    # --- CALCULATE AREA ---
    total_disease_area = 0
    
    # Check if any masks (disease areas) were detected
    if results[0].masks is not None and len(results[0].masks) > 0:
        # Get the original image dimensions (height, width)
        h, w = results[0].orig_shape
        image_area = h * w
        
        # Loop through each detected mask
        for mask in results[0].masks:
            # The mask data contains the polygon points.
            # We can approximate the area by counting the number of non-zero pixels in the mask's binary representation.
            # Ultralytics provides a convenient way to get the binary mask tensor.
            # The mask tensor has shape [1, H, W]. We sum the non-zero pixels.
            total_disease_area += mask.data.sum()
            
        # Normalize the area by the total image area to get a percentage
        area_percentage = (total_disease_area / image_area).item() # .item() gets the raw number from the tensor
        
        # We will output a number from 0.0 to 1.0
        print(f"{area_percentage:.4f}")
        
    else:
        # If no disease is found, the area is 0
        print(0.0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect disease in an image and calculate the affected area.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    args = parser.parse_args()
    predict_disease_area(args.image)