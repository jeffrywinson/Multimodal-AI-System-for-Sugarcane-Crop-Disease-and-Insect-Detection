# create_list.py
import os

# --- Configuration ---
# The directory where your combined images are located
IMAGE_DIR = os.path.join('datasets', 'insect_data_combined', 'images', 'train')

# The output path for the new train.txt file
OUTPUT_FILE = os.path.join('datasets', 'insect_data_combined', 'train.txt')

# --- Main Script ---
def create_file_list():
    """
    Scans the image directory and creates a train.txt file with relative paths.
    """
    print(f"📝 Creating new train.txt file for directory: {IMAGE_DIR}")
    
    # Get all image files from the directory
    try:
        image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    except FileNotFoundError:
        print(f"❌ ERROR: Directory not found at {IMAGE_DIR}. Make sure you ran the combine script first.")
        return

    # Create a list of relative paths for the txt file
    # Path format should be relative to the project root, matching YOLO's expectation
    # e.g., datasets/insect_data_combined/images/train/image1.jpg
    relative_paths = [os.path.join(IMAGE_DIR, f).replace("\\", "/") for f in image_files]

    # Write the paths to the output file
    with open(OUTPUT_FILE, 'w') as f:
        for path in relative_paths:
            f.write(path + '\n')
    
    print(f"✅ Successfully created {OUTPUT_FILE} with {len(relative_paths)} entries.")

if __name__ == '__main__':
    create_file_list()