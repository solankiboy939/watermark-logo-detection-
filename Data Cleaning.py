import os
dataset_dir = r"C:\Users\shiva\OneDrive\Desktop\solanki\WatermarkDataset\Data"
image_extensions = (".jpg", ".jpeg", ".png")
deleted_count = 0
for filename in os.listdir(dataset_dir): 
    if not filename.endswith(".txt"): 
        continue

    label_path = os.path.join(dataset_dir, filename)
    with open(label_path, "r") as f:
        lines = f.readlines()

    remove = False
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            remove = True
            break
        try:
            _, x, y, w, h = map(float, parts)
            if any(val < 0 or val > 1 for val in [x, y, w, h]):
                remove = True
                break
        except ValueError:
            remove = True
            break

    if remove:
        # Delete label file
        os.remove(label_path)

        # Try deleting matching image
        image_name = os.path.splitext(filename)[0]
        deleted_image = False
        for ext in image_extensions:
            image_path = os.path.join(dataset_dir, image_name + ext)
            if os.path.exists(image_path):
                os.remove(image_path)
                deleted_image = True
                break

        deleted_count += 1
        print(f"🗑️ Deleted corrupt label and image: {filename} + image")
print(f"Deleted {deleted_count} images and labels")
