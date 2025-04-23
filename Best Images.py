import cv2
import os

def calculate_sharpness(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0
    return cv2.Laplacian(image, cv2.CV_64F).var()

def select_best_images(input_folder, output_folder, top_n = 10000):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    sharpness_scores = []
    image_extensions = ('.jpg', '.jpeg', '.png')

    files = [f for f in os.listdir(input_folder) if f.lower().endswith(image_extensions)]
    for f in files:
        img_path = os.path.join(input_folder,f)
        if os.path.isfile(img_path):
            sharpness = calculate_sharpness(img_path)
            sharpness_scores.append((sharpness, f))
    sharpness_scores.sort(key = lambda x:x[0])
    for i in range(top_n):
        score = sharpness_scores[i][0]
        img = sharpness_scores[i][1]
        out_path = os.path.join(output_folder,img)
        inp_path = os.path.join(input_folder,img)
        cv2.imwrite(out_path, cv2.imread(inp_path))
        print(f"Saved image {img} successfully")
    print("Top 10000 images saved successfully")

if __name__=="__main__":
    input_folder = r"C:\Users\shiva\OneDrive\Desktop\solanki\WatermarkDataset\images\train"
    output_folder = r"C:\Users\shiva\OneDrive\Desktop\solanki\WatermarkDataset\Data"
    select_best_images(input_folder, output_folder, top_n=10000)


        

