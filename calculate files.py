import os

# Set your folder path here
folder_path = r"C:\Users\shiva\OneDrive\Desktop\prabh\Dataset\Without Mask"

# Initialize counters
jpg_count = 0
txt_count = 0

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(".jpg"):
        jpg_count += 1
    elif filename.lower().endswith(".txt"):
        txt_count += 1

print(f"Total JPG files: {jpg_count}")
print(f"Total TXT files: {txt_count}")
