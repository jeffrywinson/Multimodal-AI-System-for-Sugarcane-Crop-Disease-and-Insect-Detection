# train_yolov8s_insect.py
# This script executes the training process for the YOLOv8s Insect Detection model
# on a local machine with a CUDA-enabled GPU.

from ultralytics import YOLO
import os

# --- Local Configuration Parameters ---
# Define the path to your data.yaml file on your local Windows machine
# Make sure this path is correct for your local setup!
data_path_local = 'datasets/insect_detection_data.yaml' # Relative path from C:\Hackathon_Project

# Define the project and name for saving results locally
# Results will be saved inside C:\Hackathon_Project\runs\detect\insect_training_local\
project_dir_local = 'runs/detect/insect_training_local'
run_name_local = 'yolov8s_insect_aug_model_local_run'

# --- Training Hyperparameters (Consistent with Colab run) ---
epochs_val = 500
patience_val = 50
imgsz_val = 640
batch_size_val = 16

# --- Augmentation Parameters (Consistent with Colab run) ---
fliplr_val = 0.5
mosaic_val = 1.0
mixup_val = 0.1
degrees_val = 15
translate_val = 0.1
scale_val = 0.9
shear_val = 5
perspective_val = 0.0002
hsv_h_val = 0.015
hsv_s_val = 0.7
hsv_v_val = 0.4

print("--- Starting YOLOv8s Insect Detection Training (Local Execution) ---")
print(f"Data YAML: {data_path_local}")
print(f"Results will be saved to: {os.path.join(project_dir_local, run_name_local)}")
print(f"Epochs: {epochs_val}, Patience: {patience_val}")
print(f"Image Size: {imgsz_val}, Batch Size: {batch_size_val}")
print("Augmentations enabled: Fliplr, Mosaic, Mixup, Geometric, Photometric (HSV)")
print("------------------------------------------------------------------")

# Load a pre-trained YOLOv8s model
# It will automatically download 'yolov8s.pt' if not found in cache.
print("Loading YOLOv8s model...")
model = YOLO('yolov8s.pt') # Assumes yolov8s.pt is in the working directory or Ultralytics cache

# Ensure CUDA is available before training
if torch.cuda.is_available():
    print(f"CUDA is available! Training on: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA is NOT available. Training will proceed on CPU, which will be much slower.")

# Train the model with specified parameters
results = model.train(
    data=data_path_local,
    epochs=epochs_val,
    patience=patience_val,
    imgsz=imgsz_val,
    batch=batch_size_val,
    fliplr=fliplr_val,
    mosaic=mosaic_val,
    mixup=mixup_val,
    degrees=degrees_val,
    translate=translate_val,
    scale=scale_val,
    shear=shear_val,
    perspective=perspective_val,
    hsv_h=hsv_h_val,
    hsv_s=hsv_s_val,
    hsv_v=hsv_v_val,
    project=project_dir_local,
    name=run_name_local,
    # device=0 # Optional: Specify GPU device index if you have multiple GPUs, 0 is usually default
)

print("\n--- YOLOv8s Insect Detection Training Completed (Local) ---")
print(f"Best model saved to: {os.path.join(project_dir_local, run_name_local, 'weights', 'best.pt')}")
print("---------------------------------------------------------")

# Optional: You can add a simple prediction test here
# print("\n--- Testing inference on a sample image (Optional) ---")
# # Make sure you have a test image available locally, e.g., in datasets/insect_detection_data/images/val/
# test_image_path = 'datasets/insect_detection_data/images/val/your_test_image.jpg' # REPLACE with an actual image path
# if os.path.exists(test_image_path):
#     results_inference = model(test_image_path) # Predict on the test image
#     for r in results_inference:
#         im_bgr = r.plot() # plot a BGR numpy array of predictions
#         import cv2
#         cv2.imwrite('test_image_prediction.jpg', im_bgr) # Save the predicted image
#         print(f"Prediction saved to test_image_prediction.jpg")
# else:
#     print(f"Test image not found at {test_image_path}")