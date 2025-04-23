import cv2
import os
from cvzone.FaceDetectionModule import FaceDetector

# Settings
confidence = 0.5
offsetPercentageW = 10
offsetPercentageH = 10
floatingPoint = 6
classID = 1
save = True

# Paths
outputFolderPath = r"C:\Users\shiva\OneDrive\Desktop\prabh\Dataset\Without Mask"
directory = r"C:\Users\shiva\OneDrive\Desktop\prabh\Dataset\Without Mask"

# Initialize detector
detector = FaceDetector()

# Get only image files
image_extensions = ('.jpg', '.jpeg', '.png')
files = [f for f in os.listdir(directory) if f.lower().endswith(image_extensions)]

for f in files:
    image_path = os.path.join(directory, f)

    if not os.path.isfile(image_path):
        continue

    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Skipping invalid image: {f}")
        os.remove(image_path)
        continue

    faces, bboxs = detector.findFaces(img, draw=False)
    listInfo = []

    if bboxs:
        for i, faceInfo in enumerate(bboxs):
            if i == 0 and len(bboxs) > 1:
                continue

            x, y, w, h = faceInfo["bbox"]
            score = faceInfo["score"][0]

            if float(score) > confidence:
                offsetW = (offsetPercentageW / 100) * w
                x = max(int(x - offsetW), 0)
                w = max(int(w + offsetW * 2), 1)

                offsetH = (offsetPercentageH / 100) * h
                y = max(int(y - offsetH * 3), 0)
                h = max(int(h + offsetH * 4.5), 1)

                imgH, imgW, _ = img.shape
                xcenter = x + w / 2
                ycenter = y + h / 2
                xnorm = round(xcenter / imgW, floatingPoint)
                ynorm = round(ycenter / imgH, floatingPoint)
                wnorm = round(w / imgW, floatingPoint)
                hnorm = round(h / imgH, floatingPoint)

                listInfo.append(f"{classID} {xnorm} {ynorm} {wnorm} {hnorm}\n")

    if save and listInfo:
        label_filename = os.path.splitext(f)[0] + ".txt"
        label_path = os.path.join(outputFolderPath, label_filename)
        with open(label_path, "w") as label_file:
            label_file.writelines(listInfo)
        print(f"✅ Saved labels for: {f}")
    else:
        print(f"❌ No valid face detected in: {f}. Deleting image.")
        os.remove(image_path)

print("🎉 All valid bounding box data saved. Invalid images removed.")
