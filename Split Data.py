import os
import random
import shutil
from itertools import islice

# Paths
outputFolderPath = r"C:\Users\shiva\OneDrive\Desktop\solanki\WatermarkDataset\SplitData"
inputFolderPath = r"C:\Users\shiva\OneDrive\Desktop\solanki\WatermarkDataset\Data"
splitRatio = {"train": 0.8, "val": 0.2}  # Only train and val for YOLOv5

# Clean output folder if it exists
if os.path.exists(outputFolderPath):
    shutil.rmtree(outputFolderPath)
    print("Removed existing SplitData directory")

# Create necessary folders
for split in ["train", "val"]:
    os.makedirs(f"{outputFolderPath}/{split}/images", exist_ok=True)
    os.makedirs(f"{outputFolderPath}/{split}/labels", exist_ok=True)

# Get all unique base filenames (without extensions)
all_files = os.listdir(inputFolderPath)
unique_names = list(set([f.split('.')[0] for f in all_files]))
random.shuffle(unique_names)

# Split data
len_data = len(unique_names)
len_train = int(len_data * splitRatio["train"])
len_val = len_data - len_train

train_set = unique_names[:len_train]
val_set = unique_names[len_train:]

print(f"Total Files: {len_data}")
print(f"Split - Train: {len(train_set)}, Val: {len(val_set)}")

# Copy matching image-label pairs
for split_name, split_data in zip(["train", "val"], [train_set, val_set]):
    for base in split_data:
        img_path = os.path.join(inputFolderPath, base + ".jpg")
        label_path = os.path.join(inputFolderPath, base + ".txt")

        # Only copy if both image and label exist
        if os.path.exists(img_path) and os.path.exists(label_path):
            shutil.copy(img_path, os.path.join(outputFolderPath, split_name, "images", base + ".jpg"))
            shutil.copy(label_path, os.path.join(outputFolderPath, split_name, "labels", base + ".txt"))

print("Split process completed.")

# Create data.yaml for YOLOv5
classes = ['watermark']  # Only one class for watermark detection
data_yaml = f"""train: {outputFolderPath}/train/images
val: {outputFolderPath}/val/images

nc: {len(classes)}
names: {classes}
"""

with open(os.path.join(outputFolderPath, "data.yaml"), "w") as f:
    f.write(data_yaml)

print("data.yaml file created for watermark detection.")
