import os
from PIL import Image

# --- 1. Configuration: UPDATE THIS PATH ---
# IMPORTANT: Provide the full path to the folder containing the images you want to resize.
folder_path = r"C:\Users\jeffr\Desktop\Team14\lv6" 

# Define the target resolution
target_size = (1024, 1024)

# --- 2. The Resizing Logic ---
print(f"Starting to resize images in: {folder_path}")

# Check if the directory exists
if not os.path.isdir(folder_path):
    print(f"❌ Error: The folder path '{folder_path}' does not exist. Please check the path.")
else:
    # Get a list of all files in the directory
    files = os.listdir(folder_path)
    
    # Filter for image files only
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    processed_count = 0

    for filename in image_files:
        try:
            image_path = os.path.join(folder_path, filename)
            
            # Open the image file
            with Image.open(image_path) as img:
                # Resize the image. Using Image.Resampling.LANCZOS ensures high quality.
                resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                
                # Save the resized image, overwriting the original file
                resized_img.save(image_path)
                
                print(f"Resized '{filename}'")
                processed_count += 1

        except Exception as e:
            print(f"❌ Could not process file {filename}. Error: {e}")
        
    print(f"\n✅ Done! Resized {processed_count} images to {target_size}.")