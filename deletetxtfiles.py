import os

# Set your folder path here
folder_path = r"C:\Users\shiva\OneDrive\Desktop\solanki\WatermarkDataset\Data"

# Get all file names in the folder
files = os.listdir(folder_path)

# Create a set of JPG file base names (without extension)
jpg_basenames = {os.path.splitext(f)[0] for f in files if f.lower().endswith(".jpg")}

# Loop through all .txt files and delete if no matching .jpg file exists
deleted_count = 0
for f in files:
    if f.lower().endswith(".txt"):
        base = os.path.splitext(f)[0]
        if base not in jpg_basenames:
            txt_path = os.path.join(folder_path, f)
            os.remove(txt_path)
            print(f"🗑️ Deleted: {f}")
            deleted_count += 1

print(f"\n✅ Total TXT files deleted: {deleted_count}")
