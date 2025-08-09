# combine_files.py
import os
import shutil

# --- Configuration ---
DEST_DIR = os.path.join('datasets', 'insect_data_combined')
SOURCE_DIRS = [
    r'C:\Users\shann\OneDrive\Desktop\l514',  # ❗️ UPDATE THIS PATH
    r'C:\Users\shann\OneDrive\Desktop\l539'   # ❗️ UPDATE THIS PATH
]

# --- Main Script ---
def combine_unique_datasets():
    print("🚀 Combining unique dataset files...")

    # Create destination directories
    dest_img_path = os.path.join(DEST_DIR, 'images', 'train')
    dest_lbl_path = os.path.join(DEST_DIR, 'labels', 'train')
    os.makedirs(dest_img_path, exist_ok=True)
    os.makedirs(dest_lbl_path, exist_ok=True)

    total_files_copied = 0
    # Process each source directory
    for source_path in SOURCE_DIRS:
        print(f"\nProcessing source: '{source_path}'")
        source_img_path = os.path.join(source_path, 'images', 'train')

        if not os.path.exists(source_img_path):
            print(f"⚠️ Warning: Image directory not found at {source_img_path}. Skipping.")
            continue

        image_files = [f for f in os.listdir(source_img_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for image_file in image_files:
            # Copy image
            shutil.copy2(os.path.join(source_img_path, image_file), dest_img_path)

            # Copy corresponding label
            label_file = os.path.splitext(image_file)[0] + '.txt'
            source_label_file = os.path.join(source_path, 'labels', 'train', label_file)
            if os.path.exists(source_label_file):
                shutil.copy2(source_label_file, dest_lbl_path)

        total_files_copied += len(image_files)
        print(f"  - Copied {len(image_files)} images and their labels.")

    print(f"\n🎉 Combination complete! Total images in combined folder: {total_files_copied}")

if __name__ == '__main__':
    combine_unique_datasets()