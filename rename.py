import os

# --- 1. Configuration: UPDATE THIS PATH ---
# IMPORTANT: Provide the full path to the folder containing your JPG images.
folder_path = r'C:\Users\jeffr\Desktop\Team14\lv6' 

# Define the new base name for your files
new_base_name = 'lv6_t14'

# --- 2. The Renaming Logic ---
print(f"Starting to rename files in: {folder_path}")

# Check if the directory exists
if not os.path.isdir(folder_path):
    print(f"❌ Error: The folder path '{folder_path}' does not exist. Please check the path.")
else:
    # Get a list of all files and filter for .jpg files
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
    
    # Sort the files to ensure a consistent renaming order
    image_files.sort()
    
    counter = 1
    renamed_count = 0

    for filename in image_files:
        # Create the new filename
        new_filename = f"{new_base_name}_{counter}.jpg"
        
        # Get the full path for the old and new names
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)
        
        # Rename the file
        os.rename(old_path, new_path)
        
        print(f"Renamed '{filename}' to '{new_filename}'")
        
        counter += 1
        renamed_count += 1
        
    print(f"\n✅ Done! Renamed {renamed_count} image files.")