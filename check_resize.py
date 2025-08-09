import os
from PIL import Image

# --- 1. Configuration: UPDATE THIS PATH ---
# IMPORTANT: Provide the full path to the folder containing the images to check.
folder_path = r"C:\Users\jeffr\Desktop\Team14\dh7" 

# Define the expected resolution
expected_size = (1024, 1024)

# --- 2. The Dimension Checking Logic ---
print(f"Starting to check image dimensions in: {folder_path}")

# A list to store files with incorrect sizes
mismatched_files = []

# Check if the directory exists
if not os.path.isdir(folder_path):
    print(f" Error: The folder path '{folder_path}' does not exist. Please check the path.")
else:
    # Get a list of all image files in the directory
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for filename in image_files:
        try:
            image_path = os.path.join(folder_path, filename)
            
            # Open the image file to read its properties
            with Image.open(image_path) as img:
                # Check if the image size matches the expected size
                if img.size != expected_size:
                    mismatched_files.append((filename, img.size)) # Store filename and its actual size

        except Exception as e:
            print(f"Could not process file {filename}. Error: {e}")

# --- 3. Report the Results ---
if not mismatched_files:
    print(f"\n Success! All {len(image_files)} images have the correct {expected_size} resolution.")
else:
    print(f"\n Found {len(mismatched_files)} images with incorrect dimensions:")
    for filename, size in mismatched_files:
        print(f"  - File: {filename}, Actual Size: {size[0]}x{size[1]}")