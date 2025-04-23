import os
import glob

# Set the folder path
folder_path = r"C:\Users\shiva\OneDrive\Desktop\prabh\Dataset\Without Mask"  # change this to your target folder

# Find all .txt files in the folder
txt_files = glob.glob(os.path.join(folder_path, "*.txt"))

# Delete each .txt file
for file_path in txt_files:
    try:
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")
